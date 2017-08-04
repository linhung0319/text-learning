#!/usr/bin/env python3

from nltk.stem.snowball import SnowballStemmer

def parseOutText(f)
	"""
	   given an opened email file f, parse out all text
	   below the metadata block at the top 

	   add stemming capabilities and return a string
	   that contains all the words in the email

	   example use case:
	   f = open("email_file_name.txt", "r")
	   text = parseOutText(f)
	"""

def main()
	f = open("test_email.txt", "r")
	text = parseOutText(f)
	print text

if __name__ == 'main':
	main()
