import pandas as pd 
import numpy as np 
import json
import nltk
import re,csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


%matplotlib inline
pd.set_option('display.max_colwidth',300)

meta = pd.read_csv("Movie Genre Prediction- Multi Class Label/movie.metadata.tsv",sep='\t',header=None)

meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

plots = []
with open("Movie Genre Prediction- Multi Class Label/plot_summaries.txt",'r') as f:
    reader = csv.reader(f,dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)


plot = []
movie_id = []

for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

movies = pd.DataFrame({'movie_id':movie_id,'plot':plot})

print(movies.head())

meta['movie_id'] = meta['movie_id'].astype(str)

movies = pd.merge(movies,meta[['movie_id','movie_name','genre']],on='movie_id')

print(movies.head())


genres = []

for i in movies['genre']:
    genres.append(list(json.loads(i).values()))

movies['genre_new'] = genres
movies_new = movies[~(movies['genre_new'].str.len()==0)]

all_genres = nltk.FreqDist(all_genres)

all_genres_df = pd.DataFrame({'Genre':list(all_genres.keys()),
                                'Count':list(all_genres.values())})
            

g = all_genres_df.nlargest(columns="Count",n=50)
plt.figure(figsize=(12,15))
ax = sns.barplot(data=g,x='Count',y="Genre")
ax.set(ylabel='Count')
plt.show()

def clean_text(text):

    text = re.sub("\'","",text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = ' '.join(text.split())
    text = text.lower()

    return text

movies_new['clean_plot'] = movies_new['plot'].apply(lambda x:clean_text(x))


def freq_words(x,terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdlist = nltk.FreqDist(all_words)

    words_df = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})


    d = words_df.nlargest(columns="count",n=terms)

    plt.figure(figsie=(12,15))
    ax = sns.barplot(data=d,x="count",y="word")
    ax.set(ylabel='Word')
    plt.show()


freq_words(movies_new['clean_plot'],100)

stop_words = set(stopwords.word('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

y = multilabel_binarizer.transform(movies_new['genre_new'])


tfidf_vectorizer = TfidfVectorizer(max_df=0.8,max_features=10000)


xtrain,xval,ytrain,yval = train_test_split(movies_new['clean_plot'],y,test_size=0.2,random_state=9)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

clf.fit(xtrain_tfidf,ytrain)

y_pred = clf.predict(xval_tfidf)


muultilabel_binarizer.inverse_transform(y_pred)[3]

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


for i in range(5):
    k = xval.sample(1).index[0]
    print("Movie: ",movies_new['movie_name'][k],"\nPredicted genre: ",infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")
