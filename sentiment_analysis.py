from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(score):
    if score >= 0.5:
        return "very positive"
    elif score >= 0.1:
        return "slightly positive"
    elif score <= -0.5:
        return "very negative"
    elif score <= -0.1:
        return "slightly negative"
    else:
        return "neutral"

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    return classify_sentiment(score)

def analyze_emotions(text):
    try:
        emotion_obj = NRCLex(text)
        scores = emotion_obj.raw_emotion_scores
        if scores:
            # Return dominant emotion(s)
            max_val = max(scores.values())
            dominant = [k for k, v in scores.items() if v == max_val]
            return ', '.join(dominant)
        return "none"
    except Exception:
        return "error"

def analyze_dataframe(df, text_column='text'):
    df = df.copy()
    if text_column in df.columns:
        df['sentiment'] = df[text_column].apply(analyze_sentiment)
        df['emotions'] = df[text_column].apply(analyze_emotions)
    else:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    return df
