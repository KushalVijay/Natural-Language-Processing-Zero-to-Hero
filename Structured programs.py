import re
import nltk

def get_text(file):
    text = open(file).read()
    text = re.sub('\s+',' ',text)
    text = re.sub(r'<.*?>',' ',text)
    return text

def freq_words(url):
    freqdist = nltk.FreqDist()
    text = nltk.clean_url(url)
    for word in nltk.word_tokenize(text):
        freqdist.inc(word.lower())
    return freqdist