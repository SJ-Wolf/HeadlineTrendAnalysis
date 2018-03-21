#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:26:05 2018

@author: Mustafa
"""
import csv
import numpy as np
import pandas as pd

def process_trend_data(file_location='input/BTC-USD.csv', min_percent_change=0, normalize_percent_change=False):
    count = 0
    history = {}
    prev = None
    df = pd.read_csv(file_location)
    print(f'Total dates: {len(df)}')
    df.set_index('Date', inplace=True)
    # df['Diff'] = df.Close.shift(-1) - df.Close
    # df['Diff'] = df.Close - df.Close.shift(1)
    df['PercChange'] = (df.Close.shift(-1) - df.Close) / df.Close * 100
    df.dropna(inplace=True)
    # df['NormDiff'] = df.Diff - df.Diff.mean()
    if normalize_percent_change:
        df['PercChange'] = df.PercChange - df.PercChange.median()
    df.where(df['PercChange'].abs() >= min_percent_change, inplace=True)
    df.dropna(inplace=True)
    df['Trend'] = np.where(df.PercChange >= 0, '1', '0')
    history = df.Trend.to_dict()
    print(f'Dates of interest: {len(history)}')
    # with open(file_location, 'r', encoding='utf8') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     next(reader)
    #     for row in reader:
    #         count += 1
    #         newrow = row[0].split(',')
    #         if prev is None:
    #             prev = float(newrow[4])
    #             continue
    #         diff = float(newrow[4])-prev
    #         # if abs(diff/prev) < min_ratio_increase:
    #         #     continue
    #         sign = "1"
    #         if diff < 0:
    #             sign = "0"
    #         history[newrow[0]] = sign
    #         prev = float(newrow[4])
    # print(count, len(history))
    print(f'Mean trend after minimum percent change: {pd.to_numeric(df.Trend).mean()}')
    title_trend = []

    # titles_df = pd.read_csv('all_titles_per_day.tsv', delimiter='\t')
    # with open('title_by_day.tsv', 'r', encoding='utf8') as f:
    with open('all_titles_per_day.tsv', 'r', encoding='utf8') as f:
    #with open('title_by_day.tsv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f, dialect='excel-tab')
        next(csv_reader)
        for date, title in csv_reader:
            title = title.replace(',', '').lower()
            # if 'bitcoin' not in title:
            #     continue
            try:
                title_trend.append((date, title, history[date]))
            except KeyError:
                pass

    df = pd.DataFrame(title_trend, columns=['date', 'title', 'trend'])
    df.to_csv('title_trend.tsv', '\t', index=False)
    print(f'Number of relavent titles: {len(title_trend)}')


if __name__ == '__main__':
    process_trend_data('input/BTC-USD.csv', min_percent_change=5, normalize_percent_change=True)
