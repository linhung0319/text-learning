#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.DEBUG,
		    format='%(name)s - %(message)s')

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
	"""
	   given an opened email file f, parse out all text
	   below the metadata block at the top 

	   add stemming capabilities and return a string
	   that contains all the words in the email

	   example use case:
	   f = open("email_file_name.txt", "r")
	   text = parseOutText(f)
	"""

	logger = logging.getLogger(parseOutText.__name__)
	### go back to the beginning of the file and read all the text
	f.seek(0)
	all_text = f.read()
	logger.debug('all_text:\n%s', all_text)
	### split off metadata
	content = all_text.split("X-FileName:")
	### if email has content
	if len(content) > 1:
		logger.debug('content:\n%s', content[1])
		### remove punctuation
		text_string = content[1].translate(str.maketrans("", "", string.punctuation))
		### replace \n, \t, \r with white space
		for i in ['\n', '\t', '\r']:
			text_string = text_string.replace(i, " ")
		logger.debug('content(remove punctuation):\n%s', text_string) 
		### split the text string into individual words, stem each word,
        	### and append the stemmed word to words (make sure there's a single
        	### space between each stemmed word)
		stemmer = SnowballStemmer('english')
		text_list = text_string.strip().split(' ')
		words = ''
		for word in text_list:
			if len(word):
				words = words + stemmer.stem(word) + ' '
	logger.debug('content(stem each word, single space between each word):\n%s', words)
	return words
def main():
	f = open("test_email.txt", "r")
	parseOutText(f)
	
if __name__ == '__main__':
	main()
