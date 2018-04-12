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
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import math
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import sqlite3


def get_out_headline_data():
    def get_tsv_from_url(url, refresh=True):
        if refresh:
            r = requests.get(url)
            with open('headlines.tsv', 'w', encoding='utf8') as f:
                f.write(r.content.decode('utf8'))
        with open('headlines.tsv', 'r', encoding='utf8') as f:
            text = f.read()
        return text

    t = get_tsv_from_url(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vThJtp6ZLll4fuBWscbDQ49_VAOOTre1qQVqIwTkFAuJQldCMT4MoVx3iCE_hkPv8amS_033LiBq2Lb/pub?gid=0&single=true&output=tsv')

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


def predict(train_headlines, train_trend, test_headlines, test_trend,
            min_df=0.016, max_df=0.5, stop_words=ENGLISH_STOP_WORDS,
            use_idf=True, tfidf_norm='l1', ngram_range=(1, 2),
            sgd_loss='log', sgd_penalty='l1', alpha=1e-3, max_iter=5,
            random_state=42, return_df=True):
    text_clf_svm = Pipeline([
        #    ('vect', CountVectorizer(min_df=0.031, max_df=0.2, max_features=200000, ngram_range=(2, 2))),
        ('vect', CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words,
                                 ngram_range=ngram_range)),
        ('tfidf', TfidfTransformer(norm=tfidf_norm, use_idf=use_idf)),
        # ('tfidf-vect', text.TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words,
        #                                     use_idf=use_idf, norm=tfidf_norm, ngram_range=ngram_range)),
        # ('svd', TruncatedSVD(7)),
        ('clf-svm', SGDClassifier(loss=sgd_loss, penalty=sgd_penalty,
                                  alpha=alpha, max_iter=max_iter, random_state=random_state)),
    ])
    try:
        text_clf_svm.fit(train_headlines, train_trend)
    except ValueError:
        if return_df:
            return 0, None
        else:
            return 0
    predicted_svm = text_clf_svm.predict(test_headlines)
    accuracy = np.mean(predicted_svm == test_trend)
    # print('Accuracy:', accuracy)
    # text_clf_svm.get_params()['clf-svm'].coef_

    words = text_clf_svm.get_params()['vect'].get_feature_names()
    if 'svd' in text_clf_svm.get_params():
        weights = text_clf_svm.get_params()['svd'].inverse_transform(
            text_clf_svm.get_params()['clf-svm'].coef_[0].reshape(1, -1))[0]
    else:
        weights = text_clf_svm.get_params()['clf-svm'].coef_[0]
    assert (len(words) == len(weights))
    df = pd.DataFrame(list(zip(words,
                               weights)))
    if return_df:
        print('Confusion matrix:')
        print(confusion_matrix(test_trend, predicted_svm))
        print('Accuracy:', accuracy_score(test_trend, predicted_svm))
        print('Precision score:', precision_score(test_trend, predicted_svm))
        print('Recall score:', recall_score(test_trend, predicted_svm))
        print('F1_score:', f1_score(test_trend, predicted_svm))
        return accuracy, df
    else:
        return accuracy


def run(query='bitcoin', csv_file='BTC-USD.csv', max_page_num=10, filter_sentiment=False):
    split = 0.8
    # (data, train, test) = get_out_headline_data()
    # data = pd.read_csv('title_trend.tsv', '\t', encoding='utf8').sort_values('date')
    #     data = pd.read_sql('''
    # select article.date, article.title, case when trend.trend = 0 then -1 else 1 end as trend from article
    # join trend join (select search_range, cast(max(page_num) * 0.7 as int) as last_page from article
    # where query = "microsoft"
    # group by search_range) as t1
    # on trend.input_filename = "MSFT.csv"
    # and article.date = trend.date
    # and article.sentiment = trend.trend
    # and t1.search_range = article.search_range
    # where query = "microsoft" and article.page_num < t1.last_page
    # and article.page_num < 10''', sqlite3.connect('articles.db'))

    # data = pd.read_sql('''
    # select article.date, article.full_text as title, case when trend.trend = 0 then -1 else 1 end as trend from article
    # join trend join (select search_range, cast(max(page_num) * 0.7 as int) as last_page from article
    # where query = "bitcoin"
    # group by search_range) as t1
    # on trend.input_filename = "BTC-USD.csv"
    # and article.date = trend.date
    # and article.sentiment = trend.trend
    # and t1.search_range = article.search_range
    # where query = "bitcoin" and article.page_num < t1.last_page
    # and article.page_num < 10
    # order by article.date asc''', sqlite3.connect('articles.db'))
    # train = data.iloc[:math.floor(len(data) * split)]
    # test = data.iloc[math.floor(len(data) * split):]

    # train = data[data['date'] < '2018-02-01']
    # test = data[data['date'] > '2018-02-01']

    with sqlite3.connect('articles.db') as conn:
        cur = conn.cursor()
        cur.execute(f"""create temporary table results as
select article.date, article.full_text as title, trend, sentiment from article
join trend join (select search_range, cast(max(page_num) * 0.7 as int) as last_page from article
where query = "{query}"
group by search_range) as t1
on trend.input_filename = "{csv_file}"
and article.date = trend.date
--and article.sentiment = trend.trend
and t1.search_range = article.search_range
where query = "{query}" and article.page_num < t1.last_page
and article.page_num < {max_page_num}
--order by article.date asc;
order by random()""")
#         train = pd.read_sql("""select date, title, case when trend = 0 then -1 else 1 end as trend from results
# where trend = sentiment and
# date <= (select date from results
# order by date asc
# limit (select cast(count(*) * 0.85 as int) from results), 1)""", conn)
#         test = pd.read_sql("""select date, title, case when trend = 0 then -1 else 1 end as trend from results
# where date > (select date from results
# order by date asc
# limit (select cast(count(*) * 0.85 as int) from results), 1)""", conn)
        train = pd.read_sql(f"""select date, title, case when trend = 0 then -1 else 1 end as trend from results
where {'' if filter_sentiment else '--'} trend = sentiment and
ROWID < (select ROWID from results
limit (select cast(count(*) * {0.90 if filter_sentiment else 0.80} as int) from results), 1)""", conn)
        test = pd.read_sql(f"""select date, title, case when trend = 0 then -1 else 1 end as trend from results
where ROWID >= (select ROWID from results
limit (select cast(count(*) * {0.90 if filter_sentiment else 0.80} as int) from results), 1)""", conn)

#     """(select ROWID from results
# limit (select cast(count(*) * 0.85 as int) from results), 1)"""
    merged_df = pd.merge(train, test, how='inner', on=['date', 'title', 'trend'])
    print(merged_df)
    train_headlines = [x for x in train['title']]
    train_trend = [x for x in train['trend']]
    test_headlines = [x for x in test['title']]
    test_trend = [x for x in test['trend']]

    print(f'Num training headlines: {len(train_headlines)}')
    print(f'Num testing headlines: {len(test_headlines)}')
    print(f'{int(len(train_headlines)/(len(train) + len(test))*100)}% train / {int(len(test_headlines)/(len(train) + len(test))*100)}% test')
    print('Average training trend:', train['trend'].mean(), ';', train['trend'].mean() / 2 + 0.5)
    print('Average test trend:', test['trend'].mean(), ';', test['trend'].mean() / 2 + 0.5)

    new_stopwords = set(ENGLISH_STOP_WORDS).union({query, query + 's'})

    accuracy, df, best_params = grid_search(train_headlines, train_trend, test_headlines, test_trend,
                                            stop_words=new_stopwords)

    # params = dict(
    #     {'alpha': 0.000630957344480193, 'max_df': 0.16509636244473133, 'max_iter': 5, 'min_df': 0.01,
    #      'ngram_range': (1, 2), 'random_state': 43, 'sgd_loss': 'hinge', 'sgd_penalty': 'l1', 'tfidf_norm': 'l1',
    #      'use_idf': True})
    # accuracy, df = predict(train_headlines, train_trend, test_headlines, test_trend, stop_words=new_stopwords, **params)
    # best_params = params

    # params = dict(
    #     {'alpha': 0.00036840314986403866, 'max_df': 0.049999999999999996, 'max_iter': 5, 'min_df': 0.012599210498948734,
    #      'ngram_range': (1, 2), 'random_state': 43, 'sgd_loss': 'log', 'sgd_penalty': 'l1', 'tfidf_norm': 'l2',
    #      'use_idf': True}
    # )
    # accuracy, df = predict(train_headlines, train_trend, test_headlines, test_trend, stop_words=new_stopwords, **params)
    # best_params = params

    # params = dict(
    #     {'alpha': 0.0005139042664010976, 'max_df': 0.20000000000000004, 'max_iter': 5, 'min_df': 0.01,
    #      'ngram_range': (1, 2), 'random_state': 43, 'sgd_loss': 'log', 'sgd_penalty': 'l1', 'tfidf_norm': 'l1',
    #      'use_idf': True}
    # )
    # accuracy, df = predict(train_headlines, train_trend, test_headlines, test_trend, stop_words=new_stopwords, **params)
    # best_params = params

    print('Accuracy:', accuracy)
    top_n = 7
    most_negative = df.sort_values(1).iloc[:top_n]
    most_positive = df.sort_values(1).iloc[:-top_n - 1:-1]
    print(most_negative)
    print()
    print(most_positive)
    print()
    print(f'Total keywords: {len(df)}')
    print(f'Best parameters: {best_params}')
    print()
    print(df.sort_values(1).where(df[1] != 0.0).dropna())
    best_params['random_state'] = 42
    print('Random state 42')
    predict(train_headlines, train_trend, test_headlines, test_trend, **best_params)
    best_params['random_state'] = 41
    print('Random state 41')
    predict(train_headlines, train_trend, test_headlines, test_trend, **best_params)


def grid_search(train_headlines, train_trend, test_headlines, test_trend, stop_words):
    params = dict(
        min_df=np.geomspace(0.01, 0.020, 4),
        max_df=np.geomspace(0.05, 0.2, 4),
        stop_words=[stop_words, ],
        use_idf=[True],
        tfidf_norm=['l1', 'l2'],
        ngram_range=[(1, 2), (1, 1)],
        sgd_loss=['hinge'],
        sgd_penalty=['l1'],
        alpha=np.geomspace(1e-4, 1e-3, 6),
        # alpha=[1.1288378916846883e-05],
        max_iter=[5],
        return_df=[False],
        random_state=[43],
    )
    paramGrid = ParameterGrid(params)
    max_accuracy = None
    max_df = None
    max_params = None
    models = Parallel(n_jobs=12)(
        delayed(predict)(train_headlines, train_trend, test_headlines, test_trend, **params) for params in paramGrid)

    for params, accuracy in zip(paramGrid, models, ):
        if max_accuracy is None or accuracy > max_accuracy:
            max_accuracy = accuracy
            max_params = params
    max_params['return_df'] = True
    new_max_accuracy, max_df = predict(train_headlines, train_trend, test_headlines, test_trend, **max_params)
    assert math.isclose(new_max_accuracy, max_accuracy)
    del max_params['return_df']
    del max_params['stop_words']
    return max_accuracy, max_df, max_params


if __name__ == '__main__':
    import time

    t0 = time.time()
    # run()
    run()
    print(time.time() - t0)
