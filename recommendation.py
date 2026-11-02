"""
Movie Recommendation System - Logic Module
Algorithm: Content-Based Filtering using CountVectorizer & Cosine Similarity

Developed for College Project: AI-Based Movie Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movie_data(filepath="movies.csv"):
    """
    Step 1: Read the movie dataset from CSV and perform basic cleaning.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Handle any potential missing values by replacing NaN with empty string
    df['genres'] = df['genres'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    
    return df


def prepare_features(df):
    """
    Step 2 & 3: Combine important text features (genres, overview, keywords)
    into a single string for each movie.
    """
    # Combine genres, overview, and keywords into a single 'tags' column
    # Converting everything to lowercase for uniform comparison
    tags = (
        df['genres'] + " " + 
        df['overview'] + " " + 
        df['keywords']
    )
    
    # Clean text: lowercase and strip extra whitespace
    return tags.str.lower()


def compute_similarity(df):
    """
    Step 4 & 5:
    - Convert movie text tags into numerical vectors using CountVectorizer.
    - Calculate pairwise Cosine Similarity between all movies.
    """
    # Prepare text features
    tags = prepare_features(df)
    
    # Initialize CountVectorizer (Bag of Words)
    # Removing common English stop words (like 'the', 'is', 'in')
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    
    # Transform text data into word count vectors
    vector_matrix = vectorizer.fit_transform(tags)
    
    # Compute Cosine Similarity matrix (shape: n_movies x n_movies)
    similarity_matrix = cosine_similarity(vector_matrix)
    
    return similarity_matrix


def get_recommendations(movie_title, df, similarity_matrix, top_n=5):
    """
    Step 6: Given a movie title, find its index, sort other movies by
    similarity score, and return top_n recommendations.
    """
    # Case-insensitive search for the movie title
    matches = df[df['title'].str.strip().str.lower() == movie_title.strip().lower()]
    
    if matches.empty:
        return None  # Movie not found
    
    # Get index of the selected movie
    movie_index = matches.index[0]
    
    # Get pairwise similarity scores for this movie with all other movies
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort movies in descending order of similarity score
    # x[1] is the similarity score
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the 0th element because that is the selected movie itself (similarity = 1.0)
    top_similar_movies = sorted_movies[1:top_n + 1]
    
    # Build list of recommended movie dictionaries
    recommended_list = []
    for idx, score in top_similar_movies:
        movie_row = df.iloc[idx]
        recommended_list.append({
            'movie_id': int(movie_row['movie_id']),
            'title': movie_row['title'],
            'genres': movie_row['genres'],
            'overview': movie_row['overview'],
            'rating': float(movie_row['rating']),
            'release_year': int(movie_row['release_year']),
            'poster_url': movie_row['poster_url'],
            'similarity_score': round(float(score) * 100, 1)  # as percentage
        })
        
    return recommended_list


def get_movie_details(movie_title, df):
    """
    Helper function to get details of a single movie.
    """
    matches = df[df['title'].str.strip().str.lower() == movie_title.strip().lower()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()

