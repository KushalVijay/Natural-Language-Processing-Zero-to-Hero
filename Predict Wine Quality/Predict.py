import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize

data = pd.read_csv("Predict Wine Quality/winemag-data_first150k.csv",index_col=False)

#print(data.describe())
#print(data.info())

#set seaborn style
sns.set(style="whitegrid")

stopwords = set(stopwords.words('english'))

#Detokenizer combined tokenized elements

detokenizer = TreebankWordDetokenizer()

def clean_desp(desc):
    desc = word_tokenize(desc.lower())
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return detokenizer.detokenize(desc)

data["cleaned_description"] = data["description"].apply(clean_desp)

word_occurence = data["clean_description"].str.split(expand=True).stack().value_counts()

total_words = sum(word_occurence)

#Plot most common words

top_words = word_occurence[:30]/total_words

ax = sns.barplot(x = top_words.values,y = top_words.index)

#Setting title 
ax.set_title("% Occcurence of Most Frequent Words")
plt.show()

def points_to_class(points):

    if points in range(80,83):
        return 0
    elif points in range(83,87):
        return 1
    elif points in range(87,90):
        return 2
    elif points in range(90,94):
        return 3
    elif points in range(94,98):
        return 4
    else:
        return 5

data["rating"] = data["points"].apply(points_to_class)


num_classes = 5
embedding_dim = 300
epochs = 50
batch_Size = 128
max_len = 100

class_weights = {0:7
                 1:1,
                 2:1,
                 3:1,
                 4:7}
                 
