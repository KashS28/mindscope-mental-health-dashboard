import streamlit as st
import pandas as pd
from api import run_fetch_ui
from sentiment_analysis import analyze_dataframe
from visualize import *
from collections import Counter
import re

# ✅ Set page layout FIRST
st.set_page_config(layout='wide')

# ✅ Light/Dark Mode Toggle
theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"], index=0)
bg_color, text_color = ("#1E1E1E", "#FAFAFA") if theme == "Dark" else ("#FFFFFF", "#000000")

st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .reportview-container .markdown-text-container {{
        color: {text_color};
    }}
    h1, h2, h3 {{
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ✅ App Title
st.title("🧠 Mental Health Sentiment Dashboard")

# ✅ Load Data
df = run_fetch_ui()

if df is not None and not df.empty:
    required_cols = {'text', 'date'}
    if required_cols.issubset(df.columns):
        df = preprocess_dataframe(df)
        df = analyze_dataframe(df)

        unique_sentiments = sorted(df['sentiment'].dropna().unique())
        selected_sentiments = st.multiselect("Filter by Sentiment:", unique_sentiments, default=unique_sentiments)

        # 1. Sentiment Over Time
        st.header("📊 Sentiment Over Time")
        st.markdown("_This chart shows how sentiment has changed over time._")
        st.plotly_chart(plot_sentiment_over_time(df, selected_sentiments), use_container_width=True)

        # 2. Sentiment Distribution
        st.header("📋 Sentiment Distribution")
        st.markdown("_Displays the overall distribution of tweet sentiments._")
        st.plotly_chart(plot_sentiment_distribution(df, selected_sentiments), use_container_width=True)

        # 3. Word Cloud
        st.header("☁️ Word Cloud of Tweets")
        st.markdown("_Highlights the most frequent meaningful words in tweets._")
        plot_wordcloud(df['text'].tolist())

        # 4. Tweet Volume Over Time
        st.header("📆 Tweet Volume Over Time")
        st.markdown("_Shows the volume of tweets posted each day._")
        st.plotly_chart(plot_tweet_volume_by_day(df), use_container_width=True)

        # 5. Top Hashtags
        st.header("📈 Trending Hashtags")
        st.markdown("_Lists the most mentioned hashtags related to mental health._")
        for tag, count in extract_hashtag_frequencies(df).most_common(10):
            st.markdown(f"**#{tag}** — {count} mentions")

        # 6. Hashtag A/B Comparison
        st.header("🔀 Compare Two Hashtags")
        st.markdown("_Compare the popularity of two hashtags over time._")
        hashtags = list(extract_hashtag_frequencies(df).keys())
        if len(hashtags) >= 2:
            col1, col2 = st.columns(2)
            tag1 = col1.selectbox("Select Hashtag 1", hashtags, index=0, key="tag1")
            tag2 = col2.selectbox("Select Hashtag 2", hashtags, index=1, key="tag2")
            if tag1 and tag2 and tag1 != tag2:
                st.plotly_chart(plot_hashtag_comparison(df, tag1, tag2), use_container_width=True)
            else:
                st.warning("Please select two different hashtags.")
        else:
            st.warning("Not enough hashtag variety for comparison.")

        # 7. Average Tweet Length
        st.header("✏️ Average Tweet Length Over Time")
        st.markdown("_Analyzes how the length of tweets changes with time._")
        st.plotly_chart(plot_avg_tweet_length(df), use_container_width=True)

        # 8. Correlation Heatmap
        st.header("🧪 Correlation Heatmap")
        st.markdown("_Reveals relationships between tweet features._")
        plot_correlation_heatmap(df)

        # 9. Top Users
        st.header("👥 Top Users by Tweet Count")
        st.markdown("_Identifies the most active users in the dataset._")
        if 'user' in df.columns:
            st.plotly_chart(plot_top_users(df), use_container_width=True)

        # 10. Hashtag Co-occurrence Network
        st.header("🌐 Hashtag Co-occurrence Network")
        st.markdown("_Visualizes connections between commonly used hashtags._")
        st.plotly_chart(plot_hashtag_network(df), use_container_width=True)

        # 11. User Profiles
        st.header("🧑‍💻 User Behavior Profiles")
        st.markdown("_Shows user-level insights such as avg length and top sentiment._")
        with st.expander("View User Profiles"):
            profiles = plot_user_profiles(df)
            st.dataframe(profiles)

        # 12. Forecasting
        st.header("📈 Tweet Volume Forecast")
        st.markdown("_Predicts the number of tweets for upcoming days._")
        st.plotly_chart(plot_tweet_forecast(df, days_ahead=7), use_container_width=True)

        # 13. Summary
        st.header("📝 Summary of Public Sentiment")
        st.markdown("_A high-level overview of sentiment balance and common discussion themes._")

        sentiment_counts = df['sentiment'].value_counts(normalize=True).round(2) * 100
        sentiment_emojis = {
            "very positive": "😄",
            "slightly positive": "🙂",
            "neutral": "😐",
            "slightly negative": "😟",
            "very negative": "😠"
        }

        for label, emoji in sentiment_emojis.items():
            if label in sentiment_counts:
                st.markdown(f"- {emoji} **{sentiment_counts[label]:.0f}%** tweets are *{label}*")

        all_words = re.findall(r"\b\w+\b", ' '.join(df['text']).lower())
        filtered_words = [word for word in all_words if not word.startswith("http") and not word.startswith("#")]
        top_keywords = Counter(filtered_words).most_common(5)
        top_keywords_str = ', '.join([kw for kw, _ in top_keywords])
        st.markdown(f"**Top Keywords:** {top_keywords_str}")

    else:
        st.error("❌ Required columns 'text' and 'date' are missing.")
else:
    st.info("⬆️ Please load or fetch data above to continue.")
