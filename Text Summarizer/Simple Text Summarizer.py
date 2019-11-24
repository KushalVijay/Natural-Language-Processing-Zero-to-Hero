import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
import nltk
import urllib.request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from heapq import nlargest
from  collections import defaultdict
import requests

url = "https://en.wikipedia.org/wiki/Machine_learning"

#request = urllib.request.urlopen(url).read().decode('utf8','ignore')
#soup = BeautifulSoup(request,'html.parser')
response = requests.get(url)

soup = BeautifulSoup(response.content,'html.parser')

text_p = soup.find_all('p')

print(text_p)

for i in range(len(text_p)):
    text += text_p[i].text

text = text.lower()

tokens = [t for t in text.split()]

#print(tokens)

clean_token = tokens[:]
#define irrelevant words that include stop words , punctuations and numbers

stopword =  set(stopwords.words('english')+list(punctuation)+list("0123456789"))

for token in tokens:
    if token in stopword:
        clean_token.remove(token)

#print(clean_token)

'''Frequency distribution of 100 most common words  called BAG OF WORDS'''

freq = nltk.FreqDist(clean_token)
top_words = []
top_words = freq.most_common(100)

#print(top_words)
'''Tokenize the web page text into Sentences'''
sentences = sent_tokenize(text)

#print(sentences)

'''Create ranking ,Higher the presence of the frequent words in the sentence,higher will be the ranking'''

ranking = defaultdict(int)
for i,sent in enumerate(sentences):
    for word in word_tokenize(sent.lower()):
        if word in freq:
            ranking[i] += freq[word]
    top_sentences = nlargest(10,ranking,ranking.get)

#print(top_sentences)

sorted_sentences = [sentences[j] for j in sorted(top_sentences)]
#print(sorted_sentences)

