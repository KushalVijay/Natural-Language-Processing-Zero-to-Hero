import pandas as pd 
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

def remove_html(text):
    soup = BeautifulSoup(text,'lxml')
    html_free = soup.get_text()
    return html_free
def remove_punctuation(text):
    no_punc = "".join([c for c in text if c not in string.punctuation])
    return no_punc
review_df = pd.read('amazon_product_reviews.csv')
print(df.shape)

reviews = review_df['customer_reviews'].str.split("//",n=4,expand=True)

print(reviews.head())

review_df['review_title'] = reviews[0]
review_df['rating'] = reviews[1]
review_df['review_date'] = reviews[2]
review_df['customer_name'] = reviews[3]
review_df['review'] = reviews[4]

review_df.drop(columns='customer_reviews',inplace = True)

review_df['review'] = review_df['review'].apply(lambda x:remove_punctuation(x))
print(review_df['review'].head())

tokenizer = RegexpTokenizer(r'\w+')

review_df['review'] = review_df['review'].apply(lambda x:tokenizer.tokenize(x.lower()))

print(review_df['review'].head(20))

review_df['review'] = review_df['review'].apply(lambda x:remove_stopwords(x))
print(review_df['review'].head(10))

'''Lemmatizing'''
lemmatizer = WordNetLemmatizer()
 def word_lemmatizer(text):
     lem_text = [lemmatizer.lemmatize(i) for i in text]
     return lem_text

review_df['review'].apply(lambda x: word_lemmatizer(x))

'''Stemming'''

stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = "".join([stemmer.stem(i) for i in text])
    return stem_text

review_df['review'] = review_df['review'].apply(lambda x:=word_stemmer(x))
print(review_df['review'])


