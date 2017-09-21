#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO,
		    format='%(name)s - %(message)s')

import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(word_file='../preprocess/word_data.pkl', authors_file='../preprocess/email_authors.pkl'):
	""" 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        5 objects are returned:
            -- training/testing features
            -- training/testing labels
	    -- features name

	"""
	
	logger = logging.getLogger(preprocess.__name__)
	### the words (features) and authors (labels), already largely preprocessed
	authors_file_handler = open(authors_file, 'rb')
	email_authors = pickle.load(authors_file_handler)
	authors_file_handler.close()

	word_file_handler = open(word_file, 'rb')
	word_data = pickle.load(word_file_handler)
	word_file_handler.close()

	### test_size is the percentage of events assigned to the test set
	### (remainder go into training)
	features_train, features_test, labels_train, labels_test = train_test_split(word_data, email_authors, test_size=0.9, random_state=0)
	
	### text vectorization--go from strings to lists of numbers
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	features_train_transformed = vectorizer.fit_transform(features_train)
	features_test_transformed  = vectorizer.transform(features_test)

	### feature selection, because text is super high dimensional and can be really computationally chewy as a result
	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(features_train_transformed, labels_train)
	features_train_transformed = selector.transform(features_train_transformed).toarray()
	features_test_transformed  = selector.transform(features_test_transformed).toarray()

	### the word corresponding to each feature, after selected by SelectPercentile
	features_name = np.asarray(vectorizer.get_feature_names())
	support = np.asarray(selector.get_support())
	selected_features_name = features_name[support]

	logger.info('The number =of all emails: %s', len(labels_train) + len(labels_test))
	logger.info('The number of training emails: %s', len(labels_train))
	logger.info('The number of Chris training emails: %s', sum(labels_train))
	logger.info('The number of Sara training emails: %s', len(labels_train) - sum(labels_train))
	logger.info('The number of testing emails: %s', len(labels_test))
	logger.info('The number of Chris testing emails: %s', sum(labels_test))
	logger.info('The number of Sara testing emails: %s', len(labels_test) - sum(labels_test))
	logger.info('The number of word feature: %s', len(features_train_transformed[0]))
	
	return features_train_transformed, features_test_transformed, labels_train, labels_test, selected_features_name

def main():
	preprocess()

if __name__ == '__main__':
	main()
