import tweepy
import pandas as pd
import streamlit as st
import time
import os

# Twitter API Bearer Token
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFGZ0gEAAAAAlpcz2zjtVjMHq2WtYEVR%2FAyfaPo%3DXBnHpZ0cqUKmdg0JnHohstW4K8lTLtKWvU7t4nuRJ4P7sBXKz8"  # Replace with your actual token

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(query, max_results=100):
    try:
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=['created_at', 'text', 'author_id'],
            max_results=min(max_results, 100)
        )
        tweets = response.data if response and response.data else []
        data = [{'text': t.text, 'date': t.created_at, 'user': t.author_id} for t in tweets]
        return pd.DataFrame(data)
    except tweepy.TooManyRequests:
        st.error("Rate limit hit. Waiting 60 seconds.")
        time.sleep(60)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return pd.DataFrame()

def run_fetch_ui():
    st.subheader("📥 Choose Tweet Source")
    option = st.radio("Data Source:", ["Fetch Live Tweets", "Use Existing Dataset"])
    df = pd.DataFrame()

    if option == "Fetch Live Tweets":
        predefined_topics = {
            "Mental Health": "#mentalhealth",
            "Depression": "#depression",
            "Burnout": "#burnout",
            "Anxiety": "#anxiety",
            "Therapy": "#therapy",
            "Stress Relief": "#stressrelief",
        }
        topic = st.selectbox("Select Topic:", list(predefined_topics.keys()))
        query = predefined_topics[topic]

        if st.button("Fetch Tweets"):
            with st.spinner("Fetching tweets..."):
                df = fetch_tweets(query)
                if not df.empty:
                    df.to_csv("data/twitter_mental_health_live.csv", index=False)
                    st.success(f"{len(df)} tweets saved.")
                    st.dataframe(df)
                else:
                    st.warning("No tweets found or rate limit hit.")
    else:
        file_path = "Mental-Health-Twitter.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                df.to_csv("data/twitter_mental_health_live.csv", index=False)
                st.success(f"Loaded {len(df)} tweets from CSV.")
            else:
                st.warning("CSV is empty.")
        else:
            st.error("Mental-Health-Twitter.csv not found.")

    return df
