from __future__ import division
import nltk,re,pprint

def get_text(file):
    text = open(file).read()
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'<.*?>',' ',text)
    return text

content = get_text("test.html")