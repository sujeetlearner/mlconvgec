#!/usr/bin/python3
import sys, nltk

while 1:
	line = sys.stdin.readline()
	if not line:
		break
	print(' '.join(nltk.word_tokenize(line.strip())))
