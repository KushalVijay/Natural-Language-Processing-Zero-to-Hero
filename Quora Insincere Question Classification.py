import numpy as np 
import pandas as pd 
import os 
import operator
from gensim.models import KeyedVectors
import re,string
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input ,Embedding,SpatialDropout1D,Bidirectional,Dense

from keras.layers import concatenate,CuDNNGRU,GlobalAveragePooling1D,GlobalMaxPool1D

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pas_sequences
import tqdm
import nltk
from nltk.corpus import stopwords

test_df = pd.read_csv('Quora Question Classification Kaggle\\test.csv')
print(test_df.head())

train_df = pd.read_csv('Quora Question Classification Kaggle\\train.csv')
print(train_df.head())

lens = train_df.question_text.str.len()
print(lens.mean(),lens.std(),lens.max())

all_df = pd.concat([train_df,test_df])

print("Total number of questions: ",all_df.shape[0])
max_features = 100000
ques_len = 72

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

train_df["question_text"] = train_df["question_text"].fillna(NAN_WORD)
test_df["question_text"] = test_df["question_text"].fillna(NAN_WORD)

sub = test_df[['qid']]

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])'])')

def clean_text(s):
    return re_tok.sub(r' \1 ',s).lower()

def clean_numbers(x):
    x = re.sub('[0-9]{5,}','#####',x)
    x = re.sub('[0-9]{4}','####',x)
    x = re.sub('[0-9]{3}','###',x)
    x = re.sub('[0-9]{2}','##',x)
    return x

print("Cleaning train questions")
train_df['question_text'] = train_df['question_text'].apply(clean_text)

print("Cleaning test questions")
test_df['question_text'] = test_df['question_text'].apply(clean_text)

print("Removing numbers from train questions")
train_df['question_text'] = train_df['question_text'].apply(clean_numbers)

print("Removing numbers from test questions")
test_df['question_text'] = test_df['question_text'].apply(clean_numbers)


tokenizer = Tokenizer(num_words=max_features,oov_token=UNKNOWN_WORD)

tokenizer.fit_on_texts(list(train_df["question_text"]))

train_X = tokenizer.texts_to_sequence(train_df["question_text"])
test_X = tokenizer.texts_to_sequences(test_df["question_text"])

train_X = pad_sequences(train_X,maxlen = ques_len)
test_X = pad_sequences(test_X,maxlen=ques_len)

train_y = train_df['target'].values
test_y = test_df['target'].values

embd_file = '''load file'''

def load_embed(file):
    def get_coefs(word,*arr):
        return word,np.asarray(arr,dtype='float32')
    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file,encoding='latin'))

    return embeddings_index

print("Extracting Paragram embedding")

embeddings_index = load_embed(embd_file)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(),all_embs.std()
embed_size = all_embs.shape[1]


#rebuilding embedding matrix
nb_words =min(max_features,len(tokenizer.word_index))
embedding_matrix = np.random.normal(emb_mean,emb_std,(nb_words,embed_size))

for word,i in tokenizer.word_index.items():
    if i>=max_features:
        continue
    embeddings_vector = embeddings_index.get(word)
    if embeddings_vector is not None:
        embedding_matrix[i] = embeddings_vector

'''Building Classification Model'''

input_layer = Input(shape=(ques_len,))
embeddings_layer = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(input_layer)

x = SpatialDropout1D(0.2)(embeddings_layer)
x = Bidirectional(CuDNNGRU(90,return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(90,return_sequences=True))(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPool1D()(x)

x = concatenate([avg_pool,max_pool])
x = Dense(256,activation="relu")(x)
output_layer = Dense(1,activation="sigmoid")(x)
model = Model(inputs=input_layer,outputs = output_layer)
model.compile(
    loss='binary_crossentropy',
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    metrics=['accuracy']
)

model.summary()

checkpoint = ModelCheckpoint('saved-dmodel-{acc:03f}.h5',verbose=1,monitor='val_acc',save_best_only=True,mode='auto')

model.fit(train_X,train_y,batch_size=128,validation_split=0.1,callbacks=[checkpoint],epochs=8)

preds = model.predict([test_X],batch_size=1024,verbose=1)

preds = preds.reshape((-1,1))

pred_test_y = (preds>0.5).astype(int)
sub['prediction'] = pred_test_y

sub.to_cs("submission.csv",index=False)

