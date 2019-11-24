import nltk
from nltk.util import ngrams
from collections import Counter


text = 'Programming in C# by E.Balaguruswamy'

'''N-grams'''
_1grams = ngrams(nltk.word_tokenize(text),1)
_2grams = ngrams(nltk.word_tokenize(text),2)
_3grams = ngrams(nltk.word_tokenize(text),3)

print(_1grams)
print(_2grams)
print(_3grams)
