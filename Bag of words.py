import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

data = { 'twitter':get_tweets(),'facebook':get_gb_statuses()}

vectoriser = CountVectorizer()
vec = vectoriser.fit_transform(data['twitter'].append(data['facebook']))

df = pd.DataFrame(vec.toarray().transpose(),index=vectoriser.get_feature_names())

df.columns = ['twitter','facebook']

