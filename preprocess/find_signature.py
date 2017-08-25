#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO,
		    format='%(name)s - %(message)s')

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

### The words (features) and authors (labels), already largely processed.
word_data = pickle.load(open('word_data.pkl', 'rb'))
email_authors = pickle.load(open('email_authors.pkl', 'rb'))

### test_size is the percentage of events assigned to the test set (the remainder go into training)
features_train, features_test, labels_train, labels_test = train_test_split(word_data, email_authors, test_size=0.1, random_state=0)

### Tfidf vectorization
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

logger.info('The quantity of original training Email: %s', len(features_train.toarray()))
logger.info('The quantity of word feature: %s', len(features_train.toarray()[0]))

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### Accuracy score
logger.info('The Accuracy Score of the Decision Tree Classifier: %s', accuracy_score(labels_test, pred))

### The important feature
for index, importance_value in enumerate(clf.feature_importances_):
	if importance_value > 0.01:	
		logger.info('(feature, importance value): %s', (vectorizer.get_feature_names()[index], importance_value))
