#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO,
		    format='%(name)s - %(message)s')

import os
import pickle

from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""
logger = logging.getLogger(__name__)

from_sara = open('from_sara.txt', 'r')
from_chris = open('from_chris.txt', 'r')

email_authors = []
word_data = []

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
	for path in from_person:	
		path = os.path.join('..', path[:-1])
		email = open(path, 'r')
		logger.info("path: %s", path)
            		
		### use parseOutText to extract the text from the opened email
		text_string = parseOutText(email)

		### use str.replace() to remove the email signature
		words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
		for word in words:
			text_string = text_string.replace(word, '')

            	### append the text to word_data
		word_data.append(text_string)
            	### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
		if name == "sara":
			email_authors.append(0)
		else:
			email_authors.append(1)
		email.close()

from_sara.close()
from_chris.close()

pickle.dump(word_data, open('word_data.pkl', 'wb'))
pickle.dump(email_authors, open('email_authors.pkl', 'wb'))
