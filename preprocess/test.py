#!/usr/bin/env python

import os

from_sara = open("from_sara.txt", "r")

temp_counter = 0
for path in from_sara:
	temp_counter += 1
	if temp_counter < 10:
		print(os.path.join('..', path))
