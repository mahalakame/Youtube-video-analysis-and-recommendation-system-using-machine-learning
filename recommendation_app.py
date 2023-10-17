import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv('data.csv')



def get_video_recommendations(query_description, df):
    # Fill missing descriptions with empty strings
    df['Description'] = df['Description'].fillna('')

    # Create a new DataFrame for the query description
    query_df = pd.DataFrame({'Video Title': ['Query Video'], 'Description': [query_description]})

    # Concatenate the query DataFrame with the original DataFrame
    netflix_data = pd.concat([df, query_df], ignore_index=True)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(netflix_data['Description'])

    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Index of the query video
    idx = len(netflix_data) - 1

    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 similar videos
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return recommended videos
    recommendations = netflix_data[['Video Title', 'Description']].iloc[movie_indices]
    return recommendations
# Streamlit app
st.title("Video Recommendation App")

# Input query description
query_description = st.text_area("Enter your video description:")

if st.button("Get Recommendations"):
    if query_description:
        recommendations = get_video_recommendations(query_description, df)
        st.subheader("Recommended Videos:")
        st.table(recommendations)
    else:
        st.warning("Please enter a description.")

# Run the app with 'streamlit run video_recommendations_app.py'



