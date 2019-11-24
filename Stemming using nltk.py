#Algos for stemming are Porter , Lancaster , Snowball

from nltk import SnowballStemmer

#Language attribute must be there

stemmer = SnowballStemmer("english")

for word in ['blogging','blogged','blogs']:
    print(stemmer.stem(word))

