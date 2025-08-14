# Movie Review Classification

This project demonstrates sentiment analysis on movie reviews using machine learning, specifically Naive Bayes classifiers. It uses two datasets:
- `movie_reviewdataset_50K.csv` (subset of 1000 reviews)
- `movie_review_dataset_1K.csv` (additional 1000 review dataset)

## Project Overview

The goal is to classify movie reviews as **positive** or **negative** by training and testing Naive Bayes models. Key steps include:

1. **Data Loading**: Load movie review datasets in CSV format.
2. **Data Cleaning**: 
   - Convert text to lowercase.
   - Tokenize reviews.
   - Remove stopwords.
   - Apply stemming using NLTK's Porter Stemmer.
3. **Vectorization**: Convert cleaned text reviews into numerical feature vectors using `CountVectorizer`.
4. **Train-Test Split**: Split the dataset into training and testing sets (80-20 split).
5. **Model Training and Prediction**:
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes (event model)
6. **Evaluation**:
   - Calculate accuracy scores.
   - Generate confusion matrix.
   - Compute precision, recall, and F1-score metrics.

## Technologies & Libraries Used

- Python 3.x
- Pandas
- NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn

## Setup Instructions

1. Install required libraries:
   ```bash
   pip install pandas numpy nltk scikit-learn
