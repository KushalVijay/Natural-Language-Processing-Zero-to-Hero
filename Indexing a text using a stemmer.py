


import nltk,re,pprint
from __future__ import division
class IndexedText(object):

    def __init__(self,stemmer,text):
        self.text = text
        self.stemmer = stemmer
        self.index = nltk.Index((self._stem(word),i) 
                                for (i,word) in enumerate(text))

    def concordance(self,word,width=40):
        key = self._stem(word)
        wc = width/4  #width of context
        for  i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '%*s' %(width,lcontext[-width:])
            rdisplay = '%-*s' %(width,rcontext[:width])
            print(ldisplay,rdisplay)
    
    def _stem(self,word):
        return self._stemmer.stem(word).lower()


porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter,grail)
result = text.concordance('lie')

print(result)
    