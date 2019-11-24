from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import numpy as np 
import networkx as nx 


'''Generate clean sentences'''
def read_article(file_name):
    file = open(file_name,"r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sent in article:
        print(sent)
        sentences.append(sent.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    
    return sentences


'''Similarity matrix'''

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)

    #build vector for first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] +=1
    
    #build vector for second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] +=1
    
    return 1 - cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stopwords):
    #Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))

    for first in range(len(sentences)):
        for second in range(len(sentences)):
            if first == second:
                continue 
            similarity_matrix[first][second] = sentence_similarity(sentences[first],sentences[second],stopwords)
    return similarity_matrix


'''Generate Summary method'''

def generate_summary(file_name,top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    #Step 1 - Read text and tokenize
    sentences = read_article(file_name)
    #print(sentences)

    #Step 2 - Generate Similarity Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    #print(sentence_similarity_matrix)

    #Step 3 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    #Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
     
    #print("Indexes of top rank_sentence order are ranked_sentence")

    #print(ranked_sentence)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summarize Text: \n")
    for line in summarize_text:
        print(line)


generate_summary("Text Summarizer\Article.txt")