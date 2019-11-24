from collections import Counter
import re 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

def count(docs):

    word_counts = Counter()
    appears_in = Counter()
    total_docs = len(docs)

    for doc in docs:
        word_counts.update(doc)
        appears_in.update(set(doc))

    temp = list(zip(word_counts.keys(),word_counts.values()))

    #Word and count columns
    wc = pd.DataFrame(temp,columns = ['word','count'])

    #Rank column
    wc['rank'] = wc['count'].rank(method='first',ascending=False)

    #Percent Total column
    total = wc['count'].sum()
    wc['pct_total'] = wc['count'].apply(lambda x:x/total)

    #Cumulative percent total column
    wc = wc.sort_values(by='rank')
    wc['cut_pct_total'] = wc['pct_total'].cumsum()

    #Appears in column
    t2 = list(zip(appears_in.keys(),appears_in.values()))
    ac = pd.DataFrame(t2,columns=['word','appears_in'])
    wc =ac.merge(wc,on='word')

    #Appears in percent column
    wc['appears_in_pct'] = wc['appears_in'].apply(lambda x:x/total_docs)

    return wc.sort_values(by='rank')


def word_counter(tokens):
    word_counts = Counter()
    word_counts.update(tokens)
    return word_counts

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ^0-9]','',str(text))
    return text.split()

sample_text = """A token is a sequence of characters in a document that are useful for an analytical purpose. Often, but not always individual words. 
 A document in an NLP context simply means a collection of text, this could be a tweet, a book, or anything in between."""

'''Tokenization'''
tokens = tokenize(sample_text)

#print(tokens)

'''Word Count'''
word_count = word_counter(tokens)

#print(word_count.most_common())

'''Plotting'''
x = list(word_count.keys())[:10]
y = list(word_count.values())[:10]

#plt.bar(x,y)
#plt.show()

'''Document Makeup Dataframe'''
wc = count([tokens])

print(wc.head())

'''Cumulative distribution plot'''
#sns.lineplot(x='rank',y='cut_pct_total',data=wc)

#plt.show()

'''Tree Plot'''
#For top 13 words because they are contributing 45% words in doc
wc_top13 = wc[wc['rank'] <=13]

squarify.plot(sizes=wc_top13['pct_total'],label=wc_top13['word'],alpha=0.8)

#plt.axis('off')
#plt.show()