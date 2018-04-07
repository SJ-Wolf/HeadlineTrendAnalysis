#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:26:05 2018

@author: Mustafa
"""
import csv
import numpy as np
import pandas as pd
from dateutil import parser
import sqlite3
import os.path


def get_title_trend(history, one_per_day=False):
    title_trend = []
    included_dates = set()

    # titles_df = pd.read_csv('all_titles_per_day.tsv', delimiter='\t')
    # with open('title_by_day.tsv', 'r', encoding='utf8') as f:
    with open('all_titles_per_day.tsv', 'r', encoding='utf8') as f:
        # with open('title_by_day.tsv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f, dialect='excel-tab')
        next(csv_reader)
        for date, url, title in csv_reader:
            title = title.replace(',', '').lower()
            if one_per_day:
                if date in included_dates:
                    continue
                else:
                    included_dates.add(date)
            # if 'bitcoin' not in title:
            #     continue
            try:
                title_trend.append((date, title, history[date]))
            except KeyError:
                pass

    df = pd.DataFrame(title_trend, columns=['date', 'title', 'trend'])
    df.to_csv('title_trend.tsv', '\t', index=False, encoding='utf8')
    print(f'Number of relavent titles: {len(title_trend)}')


def process_trend_data(file_location='input/BTC-USD.csv', min_percent_change=0, normalize_percent_change=False):
    df = pd.read_csv(file_location)
    df = df[df['Date'] > '2008-07-01']
    print(f'Total dates: {len(df)}')
    df.set_index('Date', inplace=True)
    # df['Diff'] = df.Close.shift(-1) - df.Close
    # df['Diff'] = df.Close - df.Close.shift(1)
    # df['PercChange'] = ((df.Close.shift(-1) + df.Close.shift(-1)) / 2 - df.Close) / df.Close * 100
    # df['PercChange'] = ((df.Close.shift(3) + df.Close.shift(-2) + df.Close.shift(-1)) / 3 - (
    #             df.Close.shift(3) + df.Close.shift(2) + df.Close.shift(1)) / 3) / (
    #                                (df.Close.shift(3) + df.Close.shift(2) + df.Close.shift(1)) / 3) * 100
    df['PercChange'] = ((df.Close.shift(0)) - (
            df.Close.shift(2) + df.Close.shift(1)) / 2) / (
                               (df.Close.shift(2) + df.Close.shift(1)) / 2) * 100
    # df['PercChange'] = (df.Close - df.Open) / df.Open * 100
    df.dropna(inplace=True)
    # df['NormDiff'] = df.Diff - df.Diff.mean()
    if normalize_percent_change:
        df['PercChange'] = df.PercChange - df.PercChange.median()
    df.where(df['PercChange'].abs() >= min_percent_change, inplace=True)
    if normalize_percent_change:
        df['PercChange'] = df.PercChange - df.PercChange.median()
    df.dropna(inplace=True)
    df['Trend'] = np.where(df.PercChange >= 0, '1', '0')
    history = df.Trend.to_dict()
    print(f'Dates of interest: {len(history)}')

    print(f'Mean trend after minimum percent change: {pd.to_numeric(df.Trend).mean()}')
    with sqlite3.connect('articles.db') as conn:
        cur = conn.cursor()
        rows = []
        for d in history:
            rows.append((os.path.basename(file_location), d, history[d]))
        cur.executemany('replace into trend values (?, ?, ?)', rows)
    # get_title_trend(history)


def process_trump_data(normalize_change=True, min_perc_points=0.2):
    df = pd.read_csv('input/approval_topline.csv', sep=',')
    df.where(df['subgroup'] == 'Voters', inplace=True)  # Adults, All polls
    df.dropna(inplace=True)
    # df = df[['timestamp']]
    df['datetime'] = df['timestamp'].apply(func=lambda x: parser.parse(x))
    df['date'] = df['datetime'].apply(func=lambda x: x.date().isoformat())
    approval_df = df.groupby('date').apply(lambda x: x['approve_estimate'].median())
    approval_df = pd.DataFrame(approval_df, columns=['approval']).sort_index()
    approval_df['diff'] = approval_df['approval'].shift(-1) - approval_df['approval']
    approval_df.dropna(inplace=True)
    approval_df.where(approval_df['diff'].abs() > min_perc_points, inplace=True)
    approval_df.dropna(inplace=True)
    if normalize_change:
        # approval_df.where(approval_df['diff'].abs() > 0.0, inplace=True)
        # approval_df.dropna(inplace=True)
        approval_df['diff'] = approval_df['diff'] - approval_df['diff'].median()
    approval_df['trend'] = np.where(approval_df['diff'] >= 0, '1', '0')
    # print(approval_df)
    history = approval_df['trend'].to_dict()
    print(f'Dates of interest: {len(history)}')

    print(f'Mean trend after minimum percent change: {pd.to_numeric(approval_df["trend"]).mean()}')
    get_title_trend(history)
    # print(df.loc[df.groupby('date').groups['2017-03-02']]['approve_estimate'].min())


if __name__ == '__main__':
    process_trend_data('input/MSFT.csv', min_percent_change=1, normalize_percent_change=False)
    process_trend_data('input/BTC-USD.csv', min_percent_change=5, normalize_percent_change=False)
    # process_trump_data()
