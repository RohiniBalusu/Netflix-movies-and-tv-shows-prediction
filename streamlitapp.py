'''import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Label Encoder for categorical features (if needed)
rating_enc = LabelEncoder()
rating_enc.fit(["G", "PG", "PG-13", "R", "TV-MA"])  # Fit the encoder with the known ratings


# Function to make predictions
def predict(input_features):
    prediction = model.predict(input_features)
    return prediction[0]  # Return the first prediction

# Streamlit UI
st.title("Netflix Movie/TV Show Prediction")
st.write("Predict the type of Netflix movie or TV show based on various features.")

# Collecting input from the user
release_year = st.number_input("Enter Release Year (e.g., 2022):", min_value=1900, max_value=2025, value=2022)
duration = st.number_input("Enter Duration (in minutes):", min_value=0, value=90)
rating = st.selectbox("Select Rating:", options=["G", "PG", "PG-13", "R", "TV-MA"])

# Encode the rating feature
encoded_rating = rating_enc.transform([rating])[0]  # Convert the rating to encoded form

# Creating a feature vector from the user input
features = np.array([[release_year, duration, encoded_rating]])  # Use encoded rating in the features

# Make prediction when the user clicks the button
if st.button("Predict Type"):
    result = predict(features)
    st.write(f"Predicted Type: {result}")'''



import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Label Encoders for categorical features (rating, genre, country, target_age, etc.)
rating_enc = LabelEncoder()
rating_enc.fit(["G", "PG", "PG-13", "R", "TV-MA"])

genre_enc = LabelEncoder()
genre_enc.fit(["Action", "Drama", "Comedy", "Documentary", "Horror", "Sci-Fi", "Thriller"])

country_enc = LabelEncoder()
country_enc.fit(["USA", "UK", "India", "Canada", "Australia"])

target_age_enc = LabelEncoder()
target_age_enc.fit(["Kids", "Teens", "Adults", "All"])

month_added_enc = LabelEncoder()
month_added_enc.fit([f"{i:02d}" for i in range(1, 13)])

director_enc = LabelEncoder()
director_enc.fit(["Director A", "Director B", "Director C"])  # Replace with actual list of directors

cast_enc = LabelEncoder()
cast_enc.fit(["Cast A", "Cast B", "Cast C"])  # Replace with actual list of cast

language_enc = LabelEncoder()
language_enc.fit(["English", "Hindi", "Spanish", "French"])  # Replace with actual list of languages

production_company_enc = LabelEncoder()
production_company_enc.fit(["Company A", "Company B", "Company C"])  # Replace with actual list of production companies

# Function to make predictions
def predict(input_features):
    prediction = model.predict(input_features)
    return prediction[0]  # Return the first prediction

# Streamlit UI
st.title("Netflix Movie/TV Show Prediction")
st.write("Predict the type of Netflix movie or TV show based on various features.")

# Collecting input from the user
release_year = st.number_input("Enter Release Year (e.g., 2022):", min_value=1900, max_value=2025, value=2022)
duration = st.number_input("Enter Duration (in minutes):", min_value=0, value=90)
rating = st.selectbox("Select Rating:", options=["G", "PG", "PG-13", "R", "TV-MA"])
genre = st.selectbox("Select Genre:", options=["Action", "Drama", "Comedy", "Documentary", "Horror", "Sci-Fi", "Thriller"])
country = st.selectbox("Select Country:", options=["USA", "UK", "India", "Canada", "Australia"])
target_age = st.selectbox("Select Target Age Group:", options=["Kids", "Teens", "Adults", "All"])
month_added = st.selectbox("Select Month Added (e.g., 01 for January):", options=[f"{i:02d}" for i in range(1, 13)])

# New inputs for director, cast, language, and production company
director = st.selectbox("Select Director:", options=["Director A", "Director B", "Director C"])
cast = st.selectbox("Select Cast:", options=["Cast A", "Cast B", "Cast C"])
language = st.selectbox("Select Language:", options=["English", "Hindi", "Spanish", "French"])
production_company = st.selectbox("Select Production Company:", options=["Company A", "Company B", "Company C"])

# Encode the categorical features
encoded_rating = rating_enc.transform([rating])[0]
encoded_genre = genre_enc.transform([genre])[0]
encoded_country = country_enc.transform([country])[0]
encoded_target_age = target_age_enc.transform([target_age])[0]
encoded_month_added = month_added_enc.transform([month_added])[0]

encoded_director = director_enc.transform([director])[0]
encoded_cast = cast_enc.transform([cast])[0]
encoded_language = language_enc.transform([language])[0]
encoded_production_company = production_company_enc.transform([production_company])[0]

# Create a feature vector from the user input (now with 11 features)
features = np.array([[release_year, duration, encoded_rating, encoded_genre, encoded_country, 
                      encoded_target_age, encoded_month_added, encoded_director, 
                      encoded_cast, encoded_language, encoded_production_company]])

# Make prediction when the user clicks the button
if st.button("Predict Type"):
    result = predict(features)
    st.write(f"Predicted Type: {result}")
