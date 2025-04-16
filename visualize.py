import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import words
import networkx as nx
nltk.download('words', quiet=True)
valid_words = set(words.words())
from prophet import Prophet
from bertopic import BERTopic
import seaborn as sns

# ------------------- Preprocessing -------------------

def preprocess_dataframe(df):
    df = df.copy()
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].dt.date
    return df

# ------------------- Visualizations -------------------

def plot_sentiment_over_time(df, sentiments=None):
    df = preprocess_dataframe(df)
    if sentiments:
        df = df[df['sentiment'].isin(sentiments)]
    sentiment_trends = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    sentiment_pivot = sentiment_trends.pivot(index='date', columns='sentiment', values='count').fillna(0)
    sentiment_pivot = sentiment_pivot.sort_index()
    if not sentiment_pivot.empty:
        fig = px.line(sentiment_pivot, x=sentiment_pivot.index, y=sentiment_pivot.columns,
                      labels={'value': 'Tweet Count', 'date': 'Date'},
                      title='Sentiment Over Time')
        return fig
    else:
        return px.line(title="Sentiment Over Time (No data available)")

def plot_sentiment_distribution(df, sentiments=None):
    if sentiments:
        df = df[df['sentiment'].isin(sentiments)]
    sentiment_counts = df['sentiment'].value_counts()
    if not sentiment_counts.empty:
        fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                     labels={'x': 'Sentiment', 'y': 'Count'},
                     title='Sentiment Distribution')
        return fig
    else:
        return px.bar(title="Sentiment Distribution (No data available)")

def plot_sentiment_pie(df):
    sentiment_counts = df['sentiment'].value_counts()
    if not sentiment_counts.empty:
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title='Sentiment Proportions')
        return fig
    else:
        return px.pie(title="Sentiment Proportions (No data available)")

def plot_wordcloud(text_list, output_path=None):
    text = ' '.join(text_list)
    if text.strip():
        words_list = re.findall(r"\b\w+\b", text.lower())
        filtered_words = [word for word in words_list
                          if word not in ENGLISH_STOP_WORDS
                          and word in valid_words
                          and not word.startswith("http")
                          and not word.startswith("#")]
        cleaned_text = ' '.join(filtered_words)
        wc = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        if output_path:
            plt.savefig(output_path)
        else:
            st.pyplot(plt)

def plot_tweet_volume_by_day(df):
    df = preprocess_dataframe(df)
    tweet_volume = df.groupby('date').size().reset_index(name='count')
    if not tweet_volume.empty:
        fig = px.area(tweet_volume, x='date', y='count',
                      title='Tweet Volume Over Time', labels={'count': 'Number of Tweets'})
        return fig
    else:
        return px.area(title="Tweet Volume Over Time (No data available)")

def plot_avg_tweet_length(df):
    df = preprocess_dataframe(df)
    df['length'] = df['text'].apply(lambda x: len(str(x)))
    avg_length = df.groupby('date')['length'].mean().reset_index(name='avg_length')
    if not avg_length.empty:
        fig = px.line(avg_length, x='date', y='avg_length',
                      title='Average Tweet Length Over Time',
                      labels={'avg_length': 'Avg Length'})
        return fig
    else:
        return px.line(title="Average Tweet Length Over Time (No data available)")

def plot_top_users(df, top_n=10):
    if 'user' not in df.columns:
        return px.bar(title="Top Users (No user data available)")
    user_counts = df['user'].value_counts().head(top_n).reset_index()
    user_counts.columns = ['user', 'tweet_count']
    if not user_counts.empty:
        fig = px.bar(user_counts, x='user', y='tweet_count',
                     title=f'Top {top_n} Users by Tweet Count')
        return fig
    else:
        return px.bar(title="Top Users (No data available)")

def plot_sentiment_by_hashtag(df, top_n=10):
    df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x)))
    exploded = df.explode('hashtags')
    grouped = exploded.groupby(['hashtags', 'sentiment']).size().reset_index(name='count')
    top_tags = grouped.groupby('hashtags')['count'].sum().nlargest(top_n).index
    filtered = grouped[grouped['hashtags'].isin(top_tags)]
    if not filtered.empty:
        fig = px.bar(filtered, x='hashtags', y='count', color='sentiment', barmode='group',
                     title='Sentiment by Top Hashtags')
        return fig
    else:
        return px.bar(title="Sentiment by Hashtag (No data available)")

def plot_monthly_sentiment(df, sentiments=None):
    df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M').astype(str)
    data = df.copy()
    if sentiments:
        data = data[data['sentiment'].isin(sentiments)]
    sentiment_trend = data.groupby(['month', 'sentiment']).size().reset_index(name='count')
    return px.line(sentiment_trend, x='month', y='count', color='sentiment',
                   title='Monthly Sentiment Trends')

def plot_hashtag_network(df, min_cooccurrence=2):
    df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x)))
    tag_pairs = Counter()

    for tags in df['hashtags']:
        if isinstance(tags, list) and len(tags) > 1:
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair = tuple(sorted((tags[i].lower(), tags[j].lower())))
                    tag_pairs[pair] += 1

    tag_pairs = {k: v for k, v in tag_pairs.items() if v >= min_cooccurrence}
    G = nx.Graph()
    for (tag1, tag2), weight in tag_pairs.items():
        G.add_edge(tag1, tag2, weight=weight)

    if G.number_of_edges() == 0:
        return px.scatter(title="No co-occurring hashtags found")

    pos = nx.spring_layout(G, k=0.5)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')

    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"{node} ({G.degree(node)} links)")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', textposition='top center',
        text=text,
        marker=dict(size=[10 + G.degree(n)*2 for n in G.nodes()], color='skyblue', line_width=2),
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Hashtag Co-occurrence Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40)
                    ))
    return fig

def plot_correlation_heatmap(df):
    df['length'] = df['text'].apply(lambda x: len(str(x)))
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
    corr = df[['length', 'word_count', 'hashtag_count']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# ------------------- Utilities -------------------

def extract_hashtag_frequencies(df):
    hashtags = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x)))
    flat_tags = [tag.lower() for sublist in hashtags for tag in sublist]
    return Counter(flat_tags)




# ----------------- TOPIC MODELING -----------------

def generate_topic_model(df, text_column='text', top_n=10):
    try:
        topic_model = BERTopic(language="english")
        topics, _ = topic_model.fit_transform(df[text_column].tolist())
        df['topic'] = topics
        summary = topic_model.get_topic_info().head(top_n)
        return summary
    except Exception as e:
        return f"Topic modeling failed: {e}"

# ----------------- USER PROFILING -----------------

def plot_user_profiles(df):
    if 'user' not in df.columns:
        return pd.DataFrame(columns=['user', 'avg_sentiment'])

    user_summary = (
        df.groupby('user')
          .agg(avg_len=('text', lambda x: x.str.len().mean()),
               avg_sentiment=('sentiment', lambda x: x.value_counts().idxmax()))
          .reset_index()
          .sort_values(by='avg_len', ascending=False)
    )
    return user_summary


# ----------------- FORECASTING -----------------

def plot_tweet_forecast(df, days_ahead=7):
    daily_counts = df.groupby('date').size().reset_index(name='y')
    daily_counts.rename(columns={'date': 'ds'}, inplace=True)

    model = Prophet()
    model.fit(daily_counts)
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    fig = px.line(forecast, x='ds', y='yhat', title='Forecast: Tweet Volume')
    return fig

# ----------------- HASHTAG A/B COMPARISON -----------------

def plot_hashtag_comparison(df, tag1, tag2):
    df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x).lower()))
    df = df.explode('hashtags')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    tag_df = df[df['hashtags'].isin([tag1.lower(), tag2.lower()])]
    tag_counts = tag_df.groupby(['date', 'hashtags']).size().reset_index(name='count')

    if not tag_counts.empty:
        fig = px.line(tag_counts, x='date', y='count', color='hashtags',
                      title=f"Hashtag Trend: #{tag1} vs #{tag2}")
        return fig
    else:
        return px.line(title="No data for selected hashtags")