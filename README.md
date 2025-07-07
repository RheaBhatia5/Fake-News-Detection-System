#Fake News Detection Using Machine Learning

A machine learning project to classify news articles as **real** or **fake** using both traditional and advanced models including Logistic Regression, Naive Bayes, LightGBM, and XGBoost.

##Description

With the rise of digital media, misinformation spreads rapidly. This project aims to detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques. It uses TF-IDF features and multiple classification models to automatically determine whether a news article is real or fake based on its content.


## Dataset

We used a custom ISOT-style Fake News Dataset, containing:

- `title`: The headline of the article
- `text`: The full news content
- `label`: 0 for Fake News, 1 for Real News

Total Records: 10,000  
Balanced: 5,000 Real and 5,000 Fake news articles


## Technologies Used

- **Python 3**
- **pandas**, **NumPy**
- **scikit-learn**
- **spaCy** (for preprocessing)
- **LightGBM**
- **XGBoost**
- **Streamlit** (for web interface)
- **joblib** (for saving/loading models)

## Models Implemented

1. **Logistic Regression** – Simple and interpretable baseline model.
2. **Naive Bayes** – Fast and good for text data.
3. **XGBoost** – Powerful tree-based ensemble method.
4. **LightGBM** – High-speed gradient boosting from Microsoft.

All models were trained on TF-IDF vectorized representations of the cleaned text data.


## Preprocessing

Preprocessing was done using spaCy and included:

- Tokenization
- Lemmatization
- Removing stopwords and punctuation
- Lowercasing all words
- TF-IDF Vectorization of cleaned text

---

## Model Performance

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 98.9%    |
| Naive Bayes         | 98.7%    |
| LightGBM            | 97.8%    |
| XGBoost             | 97.5%    |

**Logistic Regression** gave the best results on our dataset.



## How to Run the Project Locally

1. **Clone the repository:**
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
Install the dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py

