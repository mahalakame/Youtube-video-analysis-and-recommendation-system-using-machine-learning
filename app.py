import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Replace 'YOUR_API_KEY' with your actual YouTube API key
api_key = 'AIzaSyC87p5IXbzALZPVvLcXz6fKyov0Oeg1nS8'

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to fetch and store a limited number of videos based on a keyword
def fetch_and_store_videos(api_key, keyword, region_code, days_ago, output_csv):
    # Calculate the start and end dates based on the user's input or use a default of 365 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)

    # Convert dates to ISO 8601 format
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Call the YouTube API to fetch videos based on the keyword
    search_response = youtube.search().list(
        q=keyword,
        type='video',
        part='snippet',
        regionCode=region_code,
        order='viewCount',
        maxResults=50,  # Fetch the maximum 50 videos per request
        publishedAfter=start_date_str,
        publishedBefore=end_date_str,
    ).execute()

    # Create an empty list to store dictionaries
    video_data = []

    # Populate the video_data list with dictionaries
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        video_title = search_result['snippet']['title']
        description = search_result['snippet'].get('description', '')

        video_data.append({'Video Title': video_title, 'Video ID': video_id, 'Description': description})

    # Create a DataFrame from the list of dictionaries
    video_df = pd.DataFrame(video_data)

    # Save the video data to a CSV file
    video_df.to_csv(output_csv, index=False)

# Function to get video recommendations based on the selected search option
# Function to get video recommendations based on the selected search option
def get_video_recommendations(selected_option, keyword, df):
    # Fill missing descriptions with empty strings
    df['Description'] = df['Description'].fillna('')
    
    if selected_option == "Search by Title":
        # Use video titles for similarity search
        query_column = 'Video Title'
    else:
        # Use video descriptions for similarity search (default)
        query_column = 'Description'
    
    # Create a new DataFrame for the query input
    query_df = pd.DataFrame({query_column: [keyword]})

    # Concatenate the query DataFrame with the original DataFrame
    combined_data = pd.concat([df, query_df], ignore_index=True)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_data[query_column])

    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Index of the query video
    idx = len(combined_data) - 1

    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 similar videos
    sim_scores = sim_scores[1:11]
    video_indices = [i[0] for i in sim_scores]

    # Return recommended videos based on the selected option
    recommendations = combined_data.iloc[video_indices][['Video Title', 'Description']]
    return recommendations

# Streamlit app
st.sidebar.title("Video Search Settings")
query_keyword = st.sidebar.text_input("Enter your keyword (e.g., 'Large Language models'):")

# Allow the user to input the number of days ago (optional) with a default of 365
days_ago = st.sidebar.number_input("Number of Days Ago (Optional, Default: 365)", value=365)

# Allow the user to choose the search option (by title or description)
search_option = st.sidebar.radio("Select Search Option", ["Search by Description", "Search by Title"])

if st.sidebar.button("Fetch Videos and Get Recommendations"):
    if query_keyword:
        output_csv = 'video_data.csv'
        # Fetch videos based on the keyword and user-defined days ago, or use the default
        fetch_and_store_videos(api_key, query_keyword, 'IN', days_ago, output_csv)

        # Load the stored video data
        df = pd.read_csv(output_csv)

        # Get video recommendations based on the selected option
        recommendations = get_video_recommendations(search_option, query_keyword, df)

        st.title("Video Recommendation App")
        st.subheader(f"Recommended Videos (Based on {search_option}):")
        st.table(recommendations)
    else:
        st.sidebar.warning("Please enter a keyword.")
