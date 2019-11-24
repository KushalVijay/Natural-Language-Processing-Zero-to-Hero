import nltk
from nltk import FreqDist
import  pandas as pd 
import numpy as np 
import json
pd.set_option("display.max_colwidth",200)

from nltk.corpus import stopwords
import re
import spacy
import gensim
from gensim import corpora

#For Visualisation
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt 
import seaborn as sns 


#with open('Topic Modeling\\Automotic_5.json') as f:
 #   data  = json.load(f)



df  = pd.read_json('Topic Modeling/Automotive_5.json',lines=True)

print(df.head())


#Function to plot most frequent terms

def freq_words(x,terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist  = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})

    d = words_df.nlargest(columns="count",n=terms)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d,x="word",y="count")
    ax.set(ylabel='Count')
    plt.show()


freq_words(df['reviewText'])

df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#]"," ")

stop_words = stopwords.words('english')


def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

reviews = [remove_stopwords(r.split()) for r in df['reviewText']]

reviews = [r.lower() for r in reviews]

freq_words(reviews,35)

nlp = spacy.load('en',disable=['parser','ner'])

def lemmatization(texts,tags=['NOUN','ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

tokenized_reviews = pd.Series(reviews).apply(lambda x:x.split())
print(tokenized_reviews)

reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1])

reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df['reviews'] = reviews_3

freq_words(df['reviews'],35)

dictionary = corpora.Dictionary(reviews_2)

doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]


LDA = gensim.models.ldamodel.LdaModel

lda_model = LDA(corpus=doc_term_matrix,id2word=dictionary,num_topics=7,random_state=100,chunksize=1000,passes=50)


print(lda_model.print_topics())


pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model,doc_term_matrix,dictionary)

print(vis)





