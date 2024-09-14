'''
    This is a Recommendation Service
    It is responsible for showing recommendations when the /courses/recommendations API is called
    This acts as a driver from the Client API call to the handling of various utility methods to generate recommendations

    Flow of Recommendation

    User -------> calls /courses/recommendations
    API endpoint -------> calls recommend_driver() function below
    recommend_driver() --------> calls courses_data.get_courses_data() to get courses data
    recommend_driver() --------> calls dataframe.make_dataframe() to get the courses df
    recommend_driver() --------> gets filters from user profile (filters for Career path & Previous Courses domain ??? Think about implementation)
    recommend_driver() --------> gets list of skills associated from the user model
    recommend_driver() --------> passes user_skills and courses_skills to vectorizer
    recommend_driver() --------> passes the vectorized lists to the similiarity function
    Returns the list of recommended course objects/course_ids

'''
import json
import requests
from flask import jsonify
import numpy as np
import pandas as pd
import networkx as nx
from .courses_data import get_courses_data
from ..utils.vectorizer import get_vectorized_user_matrix, get_vectorized_course_matrix
from ..utils.similarity import get_cosine_similarity
from .user_data import get_current_user_data
from ..utils.dataframe import make_threefield_dataframe, sort_dataframe, make_dataframe

def get_additional_keywords(description):
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "Extract 5-10 relevant technical skills or keywords from the given course description. Respond with a comma-separated list of keywords only."
            },
            {
                "role": "user",
                "content": f"Course description: {description}"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": False,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    headers = {
        "Authorization": "Bearer <your_perplexity_api_token>",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        keywords = response.json()['choices'][0]['message']['content'].split(',')
        return [keyword.strip().lower() for keyword in keywords]
    else:
        print(f"Error in API call: {response.status_code}")
        return []

def create_course_graph(courses_df):
    G = nx.DiGraph()
    for _, course in courses_df.iterrows():
        additional_keywords = get_additional_keywords(course['course_description'])
        all_skills = set(course['skills_associated'] + additional_keywords)
        G.add_node(course['course_id'], name=course['course_name'], skills=list(all_skills))
        for prereq in ['prerequisite_1', 'prerequisite_2', 'prerequisite_3']:
            if course[prereq] and course[prereq] != 'None':
                G.add_edge(course[prereq], course['course_id'])
    return G

def get_course_recommendations(G, user_courses, user_skills, similarity_scores, n=10):
    recommendations = []
    for course_id, score in similarity_scores.items():
        if course_id not in user_courses:
            prereqs_met = all(prereq in user_courses for prereq in G.predecessors(course_id))
            course_skills = set(G.nodes[course_id]['skills'])
            skill_overlap = len(course_skills.intersection(user_skills)) / len(course_skills)
            combined_score = 0.7 * score + 0.3 * skill_overlap
            if prereqs_met:
                recommendations.append((course_id, combined_score))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]

def recommend_driver():
    # Get organized course data from the DB
    courses_data = get_courses_data()
    courses_df = make_dataframe(courses_data)
    
    # Create course graph with enhanced skills
    course_graph = create_course_graph(courses_df)
    
    # Update courses_df with enhanced skills
    courses_df['enhanced_skills'] = [' '.join(course_graph.nodes[course_id]['skills']) for course_id in courses_df['course_id']]
    
    # Vectorize course skills using enhanced skills
    vectorized_course_skills, course_vectorizer = get_vectorized_course_matrix(courses_df[['course_id', 'enhanced_skills']].rename(columns={'enhanced_skills': 'skills_associated'}))

    # Get user data and skills
    user_data = get_current_user_data()
    user_skills = set(json.loads(json.dumps(user_data, indent=4))['skills'])
    user_skills_combined = ", ".join(user_skills)
    vectorized_user_skills = get_vectorized_user_matrix([user_skills_combined], vectorizer=course_vectorizer)

    # Compute similarity
    similarity_scores = get_cosine_similarity(vectorized_user_skills, vectorized_course_skills)
    similarity_dict = dict(zip(courses_df['course_id'], similarity_scores))
    
    # Get user's completed courses (assuming this information is available in user_data)
    user_completed_courses = user_data.get('completed_courses', [])
    recommendations = get_course_recommendations(course_graph, user_completed_courses, user_skills, similarity_dict)
    
    # Create a dataframe with recommendations
    recommendation_df = pd.DataFrame(recommendations, columns=['course_id', 'combined_score'])
    recommendation_df = recommendation_df.merge(courses_df[['course_id', 'course_name', 'enhanced_skills']], on='course_id')
    
    # Sort and filter recommendations
    filtered_recommendations = recommendation_df[recommendation_df['combined_score'] > 0.3]
    filtered_recommendations = filtered_recommendations.sort_values('combined_score', ascending=False)
    list_of_recomms = filtered_recommendations.to_dict(orient='records')
    recomms = json.dumps(list_of_recomms, indent=4)
    return recomms