import sys
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import VotingClassifier
 #Load the dataset

df = pd.read_csv('Spam Filter/SMSSpamcollection.csv')
#print(df.head())
classes = df['classes']
#Convert class labels into binary values



encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# Store the SMS message data
text_msg = df['text_msg']

#Use regular exp to replace email address, urls, phone numbers, other numbers, symbols

#replace email address with 'emailaddr'

processed = text_msg.str.replace(r'^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$','emailaddr')

# replace urls with 'webaddress'
processes = processed.str.replace(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?','webaddress')

#replace money symbols with 'moneysymb'
processed = processed.str.replace(r'^\$?([1-9]{1}[0-9]{0,2}(\,[0-9]{3})*(\.[0-9]{0,2})?|[1-9]{1}[0-9]{0,}(\.[0-9]{0,2})?|0(\.[0-9]{0,2})?|(\.[0-9]{1,2})?)$','moneysymb')

#replace 10 digit phone number with 'phonenumber'
processed = processed.str.replace(r'^[2-9]\d{2}-\d{3}-\d{4}$','phonenumber')

#replace normal numbers with 'num'
processed = processed.str.replace(r'^\s*[+-]?\s*(?:\d{1,3}(?:(,?)\d{3})?(?:\1\d{3})*(\.\d*)?|\.\d+)\s*$','num')

#replace punctuation
processed = processed.str.replace(r'[^\w\d\s]',' ')

#replace whitespace between terms with a single space 
processed = processed.str.replace(r'[\s+]',' ')

#remove leading and trailing white space 
processed = processed.str.replace(r'^\s+|\s+?$','')

#Change the words to lower case
processed = processed.str.lower()
 
#Remove stopwords


stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

#Stemming text data

ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

#Tokenization

#Creating a Bag of words
all_words = []

for msg in processed:
    words = word_tokenize(msg)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:1500]
#define find features function

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# find features for all messages
messages = list(zip(processed,Y))

#define a seed for reproducibility

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#Call find features function for each SMS

featuresets = [(find_features(text),label) for (text,label) in messages]

training,testing = model_selection.train_test_split(featuresets,test_size = 0.25,random_state = seed)

names = ['K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','Naive Bayes','SVM Linear']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]

models = zip(names,classifiers)

#Wrap models in NLTK

for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing)*100
    print(name,accuracy)
     
#Ensemble method -Voting Classifier

names = ['K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','Naive Bayes','SVM Linear']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]

models = list(zip(names,classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models,voting='hard',n_jobs = -1))

nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble,testing)*100
print('Ensemble Accuracy ',accuracy)

#Make class label prediction for testing set

txt_features ,labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)

#Print a confusion matrix and a classification report

print(pd.DataFrame(
    confusion_matrix(labels,prediction),
    index = [['actual','actual'],['ham','spam']],
    columns = [['predicted','predicted'],['ham','spam']]
))