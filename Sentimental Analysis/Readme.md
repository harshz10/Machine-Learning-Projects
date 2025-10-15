# 💬Sentiment Analysis using Python 🧠  
> *Analyze Facebook comments and understand public emotions using VADER & NLTK in Python.*

---

## 🌟 Overview
This project performs **Sentiment Analysis** on Facebook comments using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, a rule-based sentiment analysis tool from the **NLTK** library.  

The system classifies Facebook comments into three categories:
- 😊 **Positive**
- 😐 **Neutral**
- 😡 **Negative**

This helps organizations, researchers, and marketers better understand people’s opinions and emotional responses on social media platforms.

---

## 💡 What is Sentiment Analysis?
**Sentiment Analysis** is a branch of *Natural Language Processing (NLP)* that focuses on determining whether a piece of text expresses a **positive**, **negative**, or **neutral** sentiment.  

It’s widely used in:
- 🏢 **Marketing:** Analyze customer feedback  
- 🧠 **Psychology:** Detect abnormal emotional trends  
- 🗳️ **Politics:** Measure public opinion or predict election outcomes  
- 🎓 **Education:** Evaluate student feedback to improve teaching quality  

---

## ⚙️ Installation & Setup

### 🔧 Requirements
You can install all required libraries using either **conda** or **pip**.

| Library | Description | Conda Installation | Pip Installation |
|----------|--------------|--------------------|------------------|
| **NLTK** | Natural Language Toolkit | `conda install -c anaconda nltk` | `pip install nltk` |
| **NumPy** | Scientific computing | `conda install -c conda-forge numpy` | `pip install numpy` |
| **Pandas** | Data analysis | `conda install -c anaconda pandas` | `pip install pandas` |
| **Matplotlib** | Data visualization | `conda install -c conda-forge matplotlib` | `pip install matplotlib` |


---

## 🧩 Implementation Steps

### 🪄 Step 1: Load the Dataset
```python
with open('kindle.txt', encoding='ISO-8859-2') as f:
    text = f.read()
🪄 Step 2: Tokenization
Split text into sentences and words:

python
Copy code
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(text)
words = word_tokenize(text)
🪄 Step 3: Stemming
Convert words to their root form:

python
Copy code
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
for w in words:
    print(f"Actual: {w} → Stem: {porter.stem(w)}")
🪄 Step 4: Lemmatization
Normalize words using linguistic rules:

python
Copy code
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for w in words:
    print(f"Actual: {w} → Lemma: {lemmatizer.lemmatize(w)}")
🪄 Step 5: POS Tagging
Identify word types (nouns, verbs, adjectives, etc.):

python
Copy code
import nltk
print(nltk.pos_tag(words))
🪄 Step 6: Sentiment Analysis using VADER
python
Copy code
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
with open('kindle.txt', encoding='ISO-8859-2') as f:
    for line in f.read().split('\n'):
        scores = sid.polarity_scores(line)
        print(line)
        print(scores)
📈 Example Output
text
Copy code
Input: i love my kindle

compound: 0.6369
neg: 0.0
neu: 0.323
pos: 0.677
Interpretation:

✅ Positive: 67.7%

😐 Neutral: 32.3%

❌ Negative: 0%

📊 Compound Score: +0.63 → Positive sentiment

🧠 How VADER Works
VADER (Valence Aware Dictionary and sEntiment Reasoner) uses a predefined lexicon of words rated by sentiment strength.
It returns four key metrics:

Metric	Description	Range
pos	Proportion of positive sentiment	0 → 1
neu	Proportion of neutral sentiment	0 → 1
neg	Proportion of negative sentiment	0 → 1
compound	Overall normalized sentiment score	-1 → +1

📊 Optional: Visualization
Visualize sentiment scores with Matplotlib:

python
Copy code
import matplotlib.pyplot as plt

scores = [0.63, -0.12, 0.0, 0.45]
labels = ['Positive', 'Negative', 'Neutral', 'Compound']

plt.bar(labels, scores)
plt.title('Facebook Comment Sentiment Scores')
plt.xlabel('Sentiment Type')
plt.ylabel('Score')
plt.show()
🗂️ Folder Structure
bash
Copy code
Sentiment_Analysis/
│
├── sentiment_analysis.ipynb   # Main notebook
├── kindle.txt                          # Sample dataset
├── README.md                           # Documentation


