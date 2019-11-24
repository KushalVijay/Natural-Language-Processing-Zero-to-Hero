import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

reviews = pd.read_csv('Hotel Review/Hotel_REview.csv')

sia = SentimentIntensityAnalyzer()

reviews['neg'] = reviews['neg'].apply(lambda x:sia.polatrity_scores(x)['neg'])
reviews['neu'] = reviews['neu'].apply(lambda x:sia.polatrity_scores(x)['neg'])
reviews['pos'] = reviews['pos'].apply(lambda x:sia.polatrity_scores(x)['neg'])
reviews['compound'] = reviews['compound'].apply(lambda x:sia.polatrity_scores(x)['neg'])

reviews.head()

pos_review = [ j for i,j in enumerate(reviews['reviews']) if reviews['compound'][i] > 0.2]
neu_review = [ j for i,j in enumerate(reviews['reviews']) if 0.2>=reviews['compound'][i] >= -0.2]
neg_review = [ j for i,j in enumerate(reviews['reviews']) if reviews['compound'][i] < -0.2]

print("Percentage of Positive reviews: {}%".format(len(pos_review)*100/len(reviews['review'])))
print("Percentage of Neutral reviews: {}%".format(len(neu_review)*100/len(reviews['review'])))
print("Percentage of Negative reviews: {}%".format(len(neg_review)*100/len(reviews['review'])))




