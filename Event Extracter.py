'''API used id https://newsapi.org/ .API key = 5080d17b949c473092a8c4f72e03ccf5'''

newsapi = NewsApiClient(api_key='5080d17b949c473092a8c4f72e03ccf5')

def get_past_articles(past=30):
    past_articles = dict()
    for past_days in range(1,past):
        from_day = str(datetime.now() - timedelta(days=past_days))
        to_day = str(datetime.now() - timedelta(days=past_days-1))
        past_articles.update({from_day:to_day})

    return past_articles


def get_articles(query,past=30):
    past_articles = get_past_articles(past)
    all_articles = []
    for i,j in tqdm(past_articles.items()):
        for pag in tqdm(range(1,6)):
            pag_articles = newsapi.get_everything(q=query,language='en',from_param=i,to=j,
            sort_by='relevancy',page=pag)['articles']

            if len(pag_articles)==0:
                break
            all_articles.extend(pag_articles)
    return all_articles


articles = read_pickle('data/news/paris.pickle')

titles = [article['title'] for article in articles]
dates =  [article['publishedAt'] for article in articles]
descriptions = [article['description'] for article in articles]


df = pd.DataFrame({'title':titles,'date':dates,'desc':descriptions})
df = df.drop_duplicates(subset='title').reset_index(drop=True)
df = df.dropna()

df.head()
