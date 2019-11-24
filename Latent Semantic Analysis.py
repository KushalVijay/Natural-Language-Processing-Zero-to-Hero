import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap

stop_words = stopwords.words('english')
pd.set_option("display.max_colwidth",200)


dataset = fetch_20newsgroups(shuffle=True,random_state=1,remove=('headers','footers','quotes'))

documents = dataset.data

news_df = pd.DataFrame({'document':documents})

news_df['clean_doc'] = news_df['clean_doc'].str.replace("[^a-zA-Z#]"," ")

news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))

news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

tokenized_doc = tokenized_doc.apply(lmbda x: [item for item in x if item not in stop_words])

detokenied_doc = []

for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenied_doc.append(t)


news_df['clean_doc'] = detokenied_doc

vectorizer = TfidfVectorizer(stop_words='english',max_features=1000,max_df=0.5,smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])

svd_model = TruncatedSVD(n_components=20,algorithm='randomized',n_iter=100,random_state=122)

svd_model.fit(X)

terms = vectorizer.get_feature_names()

for i,comp in enumerate(svd_model.components_):
    terms_comp = zip(terms,comp)
    sorted_terms = sorted(terms_comp,key=lambda x:x[1],reverse=True)[:7]

    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")


X_topics = svd_model.fit_transform(X)

embedding = umap.UMAP(n_neighbors=150,min_dist=0.5,random_state=12).fit_transform(X_topics)


plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0],embedding[:, 1],c=dataset.target,s=10,edgecolor='none')

plt.show()
