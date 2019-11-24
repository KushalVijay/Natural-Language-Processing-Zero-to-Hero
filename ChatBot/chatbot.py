import nltk
import numpy
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('ChatBot/chatbot.txt','r',errors = 'ignore')

raw = f.read()
raw = raw.lower()

word_tokens = nltk.word_tokenize(raw)
sent_tokens = nltk.sent_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()

def Lemtokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict(ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return Lemtokens(nltk.word_tokenize(text.lower()).translate(remove_punct_dict))


GREETING_INPUTS = ("hello","hi","Greetings","sup","What's up","hey",)

GREETING_RESPONSES = ["hi","hey","*nods*","hi there","hello","I'm glad! You are talking to me"]

def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)



def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')

    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten() 
    flat.sort()
    reg_tfidf = flat[-2]

    if(reg_tfidf==0):
        robo_response += "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response += sent_tokens[idx]
        return robo_response


flag = True
print("ROBO: My name is Robo.T will answer your queries about Chatbots.If you want to exit,type Bye!")

while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!='Bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag = Falseprint("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False 
        print("ROBO: Bye! take care..")




