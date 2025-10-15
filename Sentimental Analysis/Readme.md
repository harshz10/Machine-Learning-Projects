📊 Facebook Sentiment Analysis using Python (VADER + NLTK)
🗓️ Last Updated: April 28, 2025

Author: [Your Name]
Environment: Jupyter Notebook (Anaconda)

🧠 Overview

This project performs sentiment analysis on Facebook comments using VADER (Valence Aware Dictionary and sEntiment Reasoner) from the NLTK library.

The goal is to classify Facebook comments into Positive, Negative, or Neutral sentiments — helping organizations, companies, and researchers better understand public opinions from social media data.

💡 What is Sentiment Analysis?

Sentiment Analysis is a natural language processing (NLP) technique that helps identify the emotional tone behind textual data.
It’s widely used for:

Customer Feedback Analysis

Political Sentiment Prediction

Psychological and Emotional Health Assessment

Educational Feedback Improvement

🎯 Why Sentiment Analysis?
Domain	Application
🏢 Marketing	Understand customer opinions and improve product quality
🧠 Psychology/Medicine	Detect abnormal emotions for psychological assessment
🗳️ Politics	Predict election outcomes and measure public support
🎓 Education	Collect and analyze student feedback to improve curricula
⚙️ Installation & Setup

You can install all dependencies either using conda or pip.

🧩 1. NLTK
conda install -c anaconda nltk
# or
pip install nltk

🧮 2. NumPy
conda install -c conda-forge numpy
# or
pip install numpy

🧾 3. Pandas
conda install -c anaconda pandas
# or
pip install pandas

📊 4. Matplotlib
conda install -c conda-forge matplotlib
# or
pip install matplotlib

📂 Data Source

There are several ways to collect Facebook comments:

Using Facebook Graph API

Downloading directly from Facebook

Using a dataset from Kaggle

➡️ In this project, the dataset was downloaded from Kaggle and saved locally as kindle.txt.
You can replace it with your own dataset in .txt format.

🧰 Implementation Steps
1️⃣ Load Dataset
with open('kindle.txt', encoding='ISO-8859-2') as f:
    text = f.read()

2️⃣ Tokenization

Splitting text into words and sentences:

from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(text)
words = word_tokenize(text)

3️⃣ Stemming

Normalize words to their root form:

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
for w in words:
    print(f"Actual: {w} → Stem: {porter.stem(w)}")

4️⃣ Lemmatization

Normalize words using linguistic rules:

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for w in words:
    print(f"Actual: {w} → Lemma: {lemmatizer.lemmatize(w)}")

5️⃣ POS Tagging

Identify part of speech (adjectives, verbs, etc.):

import nltk
print(nltk.pos_tag(words))

6️⃣ Sentiment Analysis (VADER)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
with open('kindle.txt', encoding='ISO-8859-2') as f:
    for line in f.read().split('\n'):
        scores = sid.polarity_scores(line)
        print(line)
        print(scores)

📈 Sample Output
Input Text: i love my kindle

compound: 0.6369
neg: 0.0
neu: 0.323
pos: 0.677


Interpretation:

Positive: 67.7%

Neutral: 32.3%

Negative: 0%

Compound Score: +0.63 (Positive sentiment)

🧩 How VADER Works

VADER (Valence Aware Dictionary and sEntiment Reasoner) uses a sentiment lexicon — a dictionary of words associated with sentiment scores.
It outputs four metrics for each text:

pos → Positive score

neu → Neutral score

neg → Negative score

compound → Normalized sum between -1 (negative) and +1 (positive)

📘 Example Use Cases

Analyze Facebook comments about a brand or product.

Measure audience sentiment toward a political figure.

Study emotional trends in educational or psychological research.

Automate customer feedback monitoring.

🧩 Folder Structure
Facebook_Sentiment_Analysis/
│
├── facebook_sentiment_analysis.ipynb   # Main Jupyter Notebook
├── kindle.txt                          # Sample dataset
├── README.md                           # Project documentation
└── requirements.txt                    # Dependencies list

🧾 Requirements File (Optional)
nltk
numpy
pandas
matplotlib

📊 Visualization (Optional)

You can extend this notebook to visualize sentiment distribution:

import matplotlib.pyplot as plt

scores = [0.63, -0.12, 0.0, 0.45]
labels = ['Positive', 'Negative', 'Neutral', 'Compound']
plt.bar(labels, scores)
plt.title('Facebook Comment Sentiment Scores')
plt.show()

🧠 Future Improvements

Integrate with Facebook Graph API for live comment scraping.

Apply Deep Learning models (BERT, RoBERTa) for more accuracy.

Build a dashboard using Streamlit or Dash for interactive visualization.

🏁 Conclusion

This project demonstrates how VADER Sentiment Analyzer and NLTK can efficiently process and classify Facebook comments into positive, negative, or neutral sentiments.
It provides a foundational framework for understanding public perception — useful in marketing, education, politics, and research.

📧 Contact

Author: [Your Name]
Email: [your.email@example.com
]
GitHub: [github.com/yourusername]
