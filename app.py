# Load the dataset
import pandas as pd
import numpy as np
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Data preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Create Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Train a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

# Save the trained model and CountVectorizer
import joblib
joblib.dump(classifier, 'sentiment_model.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')

print("Model and CountVectorizer saved successfully!")