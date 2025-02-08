from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
movies_df = pd.read_csv('movie.csv')

# Preprocessing function to handle missing genres and clean them
def preprocess_genres(df):
    df['genres'] = df['genres'].fillna('')  # Handle missing values
    df['genres'] = df['genres'].apply(lambda x: x.replace('|', ' '))  # Replace '|' with spaces
    return df

movies_df = preprocess_genres(movies_df)

# Function to remove the year from movie titles
def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()  # Remove year (YYYY) in parentheses

# Create a new column with cleaned titles for easier search and recommendations
movies_df['clean_title'] = movies_df['title'].apply(clean_title)

# Function to get movie recommendations
def get_recommendations(movie_title, movies_df):
    movie_title_cleaned = clean_title(movie_title)  # Clean the user input
    
    # Check if the movie exists in the dataset
    matched_movies = movies_df[movies_df['clean_title'] == movie_title_cleaned]
    
    if matched_movies.empty:
        return ["Movie not found. Please try again."]

    # Get the index of the first matching movie
    idx = matched_movies.index[0]

    # Vectorize movie genres
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies_df['genres'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return cleaned movie titles (without year)
    return movies_df['clean_title'].iloc[movie_indices].tolist()

# Route to display the homepage and get recommendations
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = get_recommendations(movie_title, movies_df)
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
