import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = {'twitter':get_tweets(),'facebook':get_fb_statuses()}

vectoriser = TfidfVectorizer()

vec = vectoriser.fit_transform(data['twitter'].append(data['facebook']))

df = pd.DataFrame(vec.toarray().transpose(),index = vectoriser.get_feature_names())

df.columns = ['twitter','facebook']
