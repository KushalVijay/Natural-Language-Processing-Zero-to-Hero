import nltk
def generate_model(cfdist,word,num=15):
    for i in range(num):
        print(word,end=" ")
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

print(cfd['living'])
generate_model(cfd,'living')
