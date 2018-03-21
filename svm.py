import csv
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import io
import numpy as np
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD


def get_out_headline_data():
    def get_tsv_from_url(url, refresh=True):
        if refresh:
            r = requests.get(url)
            with open('headlines.tsv', 'w', encoding='utf8') as f:
                f.write(r.content.decode('utf8'))
        with open('headlines.tsv', 'r', encoding='utf8') as f:
            text = f.read()
        return text


    t = get_tsv_from_url('https://docs.google.com/spreadsheets/d/e/2PACX-1vThJtp6ZLll4fuBWscbDQ49_VAOOTre1qQVqIwTkFAuJQldCMT4MoVx3iCE_hkPv8amS_033LiBq2Lb/pub?gid=0&single=true&output=tsv')

    csv_reader = csv.DictReader(io.StringIO(t), delimiter='\t')


    data = []
    for i, row in enumerate(csv_reader):
        assert row['Polarity'] in ('pos', 'neg'), f'Invalid polarity: {row["Polarity"]}'
        if row['Polarity'] == 'pos':
            ground_truth = 1
        else:
            ground_truth = -1
        data.append((row['Headline'], ground_truth))

    data = pd.DataFrame(data, columns=['title', 'trend'])

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    return data, train, test

# (data, train, test) = get_out_headline_data()
data = pd.read_csv('title_trend.tsv', '\t')
train = data[data['date'] < '2017-12-31']
test = data[data['date'] > '2017-12-31']
train_headlines = [x for x in train['title']]
train_trend = [x for x in train['trend']]
test_headlines = [x for x in test['title']]
test_trend = [x for x in test['trend']]

print(f'Num training headlines: {len(train_headlines)}')
print(f'Num testing headlines: {len(test_headlines)}')
print(f'{int(len(train_headlines)/len(data)*100)}% train / {int(len(test_headlines)/len(data)*100)}% test')
advancedvectorizer = text.TfidfVectorizer(min_df=0.031, max_df=0.2, max_features=200000, ngram_range=(2, 2))

# advancedvectorizer.fit_transform(train_headlines)
new_stopwords = set(ENGLISH_STOP_WORDS).union({'bitcoin'})
text_clf_svm = Pipeline([
    #    ('vect', CountVectorizer(min_df=0.031, max_df=0.2, max_features=200000, ngram_range=(2, 2))),
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    ('tfidf-vect', text.TfidfVectorizer(min_df=0.020, max_df=0.5, stop_words=new_stopwords,
                                        use_idf=True, norm='l1', ngram_range=(1, 2))),
    #('svd', TruncatedSVD(50)),
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, max_iter=5, random_state=42)),
])
text_clf_svm.fit(train_headlines, train_trend)
predicted_svm = text_clf_svm.predict(test_headlines)
print('Accuracy:', np.mean(predicted_svm == test_trend))
# text_clf_svm.get_params()['clf-svm'].coef_

df = pd.DataFrame(list(zip(text_clf_svm.get_params()['tfidf-vect'].get_feature_names(),
                           text_clf_svm.get_params()['clf-svm'].coef_[0])))

top_n = 7
most_negative = df.sort_values(1).iloc[:top_n]
most_positive = df.sort_values(1).iloc[:-top_n - 1:-1]
print(most_negative)
print()
print(most_positive)
print()
print(f'Total keywords: {len(df)}')
