import streamlit as st
import pandas as pd
from api import run_fetch_ui
from sentiment_analysis import analyze_dataframe
from visualize import (
    preprocess_dataframe,
    plot_sentiment_over_time,
    plot_sentiment_distribution,
    plot_sentiment_pie,
    plot_wordcloud,
    plot_tweet_volume_by_day,
    plot_avg_tweet_length,
    plot_top_users,
    plot_sentiment_by_hashtag,
    plot_monthly_sentiment,
    plot_hashtag_network,
    plot_correlation_heatmap,
    extract_hashtag_frequencies,
    plot_user_profiles,
    plot_tweet_forecast,
    plot_hashtag_comparison
)
from collections import Counter
import re

# ------------------- THEME + WIDTH -------------------

mode = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"], index=0)
bg_color, text_color = ("#1E1E1E", "#FAFAFA") if mode == "Dark" else ("#FFFFFF", "#000000")

st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}
    .reportview-container {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .block-container {{
        max-width: 70%;
        margin: auto;
    }}
    h1, h2, h3 {{
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------- APP START -------------------

st.title("🧠 Mental Health Sentiment Dashboard")

df = run_fetch_ui()

if df is not None and not df.empty:
    required_cols = {'text', 'date'}
    if required_cols.issubset(df.columns):
        df = preprocess_dataframe(df)
        df = analyze_dataframe(df)

        unique_sentiments = sorted(df['sentiment'].dropna().unique())
        selected_sentiments = st.multiselect("Filter by Sentiment:", unique_sentiments, default=unique_sentiments)

        st.header("🔍 Keyword / Hashtag Explorer")
        keyword = st.text_input("Enter a keyword or hashtag")
        if keyword:
            filtered = df[df['text'].str.contains(keyword, case=False, na=False)]
            st.write(f"Found {len(filtered)} tweets containing '{keyword}'")
            st.dataframe(filtered[['date', 'text', 'sentiment']] + (['user'] if 'user' in df.columns else []))

        st.header("☁️ Word Cloud of Tweets")
        plot_wordcloud(df['text'].tolist())

        st.header("📈 Trending Hashtags")
        for tag, count in extract_hashtag_frequencies(df).most_common(10):
            st.markdown(f"**#{tag}** — {count} mentions")

        st.header("📊 Sentiment Over Time")
        st.plotly_chart(plot_sentiment_over_time(df, selected_sentiments), use_container_width=True)

        st.header("📋 Sentiment Distribution")
        st.plotly_chart(plot_sentiment_distribution(df, selected_sentiments), use_container_width=True)

        st.header("📍 Sentiment Proportions")
        st.plotly_chart(plot_sentiment_pie(df), use_container_width=True)

        st.header("📆 Tweet Volume Over Time")
        st.plotly_chart(plot_tweet_volume_by_day(df), use_container_width=True)

        st.header("✏️ Average Tweet Length Over Time")
        st.plotly_chart(plot_avg_tweet_length(df), use_container_width=True)

        if 'user' in df.columns:
            st.header("👥 Top Users by Tweet Count")
            st.plotly_chart(plot_top_users(df), use_container_width=True)

        st.header("🧵 Sentiment by Top Hashtags")
        st.plotly_chart(plot_sentiment_by_hashtag(df), use_container_width=True)

        st.header("🧪 Correlation Heatmap (Length, Words, Hashtags)")
        plot_correlation_heatmap(df)

        st.header("📅 Monthly Sentiment Trend")
        st.plotly_chart(plot_monthly_sentiment(df, selected_sentiments), use_container_width=True)

        st.header("🌐 Hashtag Co-occurrence Network")
        st.plotly_chart(plot_hashtag_network(df), use_container_width=True)



        # ------------------ NEW: User Profiles ------------------
        st.header("🧑‍💻 User Behavior Profiles")
        with st.expander("View User Profiles"):
            profiles = plot_user_profiles(df)
            st.dataframe(profiles)

        # ------------------ NEW: Tweet Forecast ------------------
        st.header("📈 Tweet Volume Forecast")
        st.plotly_chart(plot_tweet_forecast(df, days_ahead=7), use_container_width=True)

        # ------------------ NEW: Hashtag A/B Comparison ------------------
        st.header("🔀 Compare Two Hashtags")
        st.header("🔀 Compare Two Hashtags")

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


        # ------------------ Summary of Emotions ------------------
        st.header("📝 Summary of Emotions")
        with st.expander("Click to expand summary", expanded=True):
            st.markdown("### 💬 Overview of Public Sentiment")

            top_words = Counter(re.findall(r"\b\w+\b", ' '.join(df['text']).lower()))
            common_words = [word for word in top_words if word not in {"https", "t", "co", "amp"}]
            top_keywords = ', '.join(list(dict(Counter(common_words).most_common(5)).keys()))

            sentiment_counts = df['sentiment'].value_counts(normalize=True).round(2) * 100

            summary_lines = []
            for label in ['very negative', 'slightly negative', 'neutral', 'slightly positive', 'very positive']:
                if label in sentiment_counts:
                    emoji = {"very negative": "😠", "slightly negative": "😟", "neutral": "😐",
                             "slightly positive": "🙂", "very positive": "😄"}[label]
                    summary_lines.append(f"- {emoji} **{sentiment_counts[label]:.0f}%** tweets are *{label}*")

            summary_lines.append(f"\n**Top keywords:** {top_keywords}")
            for line in summary_lines:
                st.markdown(line)

    else:
        st.error("❌ Required columns 'text' and 'date' are missing.")
else:
    st.info("⬆️ Please load or fetch data above to continue.")
