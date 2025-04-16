# 🧠 Mental Health Sentiment Analysis Dashboard

A Streamlit dashboard that analyzes Twitter sentiment around mental health topics in real-time. This project uses the Twitter API, NLP models, and interactive visualizations to extract meaningful emotional insights from social media conversations.

---

## 🌐 Live App
[Click here to view the dashboard](https://mental-health-dashboard-eece5642.streamlit.app/)

---

## 🚀 Features

### 🖥️ UI & Theming
- Light/Dark mode toggle
- Calming color palette for mental health context

### 📊 Visualizations
- Line graph of sentiment trends over time
- Word clouds of tweet content
- Sentiment distribution bar charts
- Trending hashtags widget

### 🔍 Interactivity
- Enter any **hashtag** or **keyword** to view matching tweets
- Display tweet summaries using extractive summarization (powered by Hugging Face Transformers)

### 🔁 Real-time Data Collection
- Fetch up to 100 tweets using Twitter API v2
- Query customizable via command-line or future UI enhancements

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | UI and dashboard framework |
| Tweepy | Twitter API client |
| Hugging Face Transformers | Summarization model |
| TextBlob | Sentiment analysis |
| Plotly & Matplotlib | Visualizations |
| Pandas | Data manipulation |

---

## 📂 Folder Structure
```
mental-health-dashboard/
├── app.py                     # Main Streamlit application
├── api.py                     # Twitter API integration
├── sentiment_analysis.py      # Sentiment classification using TextBlob
├── visualize.py               # Visual & analytics utilities
├── data/                      # Contains collected tweet CSVs
├── outputs/                   # Generated plots (optional)
└── requirements.txt           # Required Python packages
```

---

## ✅ Setup Instructions

1. **Clone this repository**
```bash
git clone https://github.com/KashS28/mental-health-dashboard.git
cd mental-health-dashboard
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Add your Twitter API Bearer Token**
Create a `.env` file:
```
BEARER_TOKEN=your_token_here
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## 📈 Example Use Cases
- Visualize public sentiment around mental health topics
- Discover trending hashtags related to emotional well-being
- Summarize how people feel about mental health over time

---

## 🙌 Acknowledgements
- Built as part of **EECE 5642 – Data Visualization Final Project**
- Twitter Developer Platform
- Hugging Face Transformers
- Streamlit Community

---

## 👩‍💻 Made with ❤️ by Kashish Shah
