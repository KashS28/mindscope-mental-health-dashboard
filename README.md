# 🧠 MindScope: Mental Health Sentiment Analysis Dashboard

[![Streamlit App](https://img.shields.io/badge/Launch-Dashboard-brightgreen?style=flat&logo=streamlit)](https://mindscope-mental-health-dashboard.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MindScope** is a real-time sentiment analysis dashboard designed to track and visualize emotional discourse around mental health on social media. Built using Python, NLP, and interactive visualizations, the app offers powerful tools for policymakers, researchers, and mental health professionals to decode the emotional pulse of the internet.

---

## 🚀 Live Demo

👉 [Launch Dashboard](https://mindscope-mental-health-dashboard.streamlit.app/)  
👉 Try toggling between **Live Twitter Data** and the **Static Dataset** to explore various features.

---

## 🔍 Features

- 📡 **Real-time Tweet Fetching** via Twitter API v2 (rate-limit aware)
- 📁 **Static Dataset Toggle** for broader hashtag and emotion coverage
- 💬 **Sentiment Analysis** using VADER and TextBlob
- 😃 **Emotion Classification** with NRCLex (e.g., anger, joy, fear)
- 📈 **Trend Visualization** over time and forecasting with ARIMA
- 🔗 **Hashtag Co-occurrence Networks** (built with NetworkX)
- 🧠 **Keyword Exploration & Word Clouds**
- 📊 **Correlation Heatmaps**, Average Tweet Length, and Tweet Frequency
- 🔍 **Compare Two Hashtags** over time for insight differentials

---

## 📊 Visualizations

- **Sentiment Distribution** (Bar)
- **Sentiment Over Time** (Line)
- **Tweet Volume Forecast** (ARIMA)
- **Hashtag Co-occurrence Network** (Graph)
- **Emotion Distribution** (Pie & Bar)
- **Correlation Heatmap** (Seaborn)
- **Keyword Explorer & Word Cloud**
- **Tweet Metadata Analysis**

---

## 📚 Tech Stack

- **Frontend:** Streamlit
- **Data Handling:** Pandas, NumPy
- **NLP & Sentiment:** VADER, TextBlob, NRCLex, NLTK, SpaCy
- **Visualization:** Plotly, Matplotlib, Seaborn, NetworkX
- **APIs:** Tweepy (Twitter API v2)
- **Forecasting:** Statsmodels (ARIMA)

---

## 📂 Project Structure

📁 mindscope-mental-health-dashboard/ 
├── app.py # Main Streamlit app 
├── api.py # Twitter API integration 
├── data/ # Static dataset (CSV) 
├── modules/ # Preprocessing, visualization, analysis helpers 
├── requirements.txt # Dependencies └── README.md # This file


---

## 📈 Use Cases

- 🏛️ **Policy makers** can assess sentiment around public decisions.
- 🧠 **Mental health NGOs** can tailor outreach strategies in real time.
- 🔬 **Researchers** can study emotional trends and linguistic patterns.
- 🚨 **Crisis teams** can act on sudden emotional spikes online.
- 🧑‍💻 **Startups** can integrate sentiment tracking into wellness products.

---

## ⚠️ Limitations

- Twitter API only returns **100 tweets per request**, so we use a **static dataset fallback** to enable full functionality.
- Hashtag-specific insights may vary between real-time and historical data modes.

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/KashS28/mindscope-mental-health-dashboard.git
   cd mindscope-mental-health-dashboard

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. (Optional) Add your Twitter API keys in api.py.

4. Run the app:
   ```bash
   streamlit run app.py



   
