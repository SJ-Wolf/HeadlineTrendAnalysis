#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:26:05 2018

@author: Mustafa
"""
import csv
import pandas as pd

count = 0
history = {}
prev = None
# with open('MSFT.csv', 'r', encoding='utf8') as csvfile:
with open('BTC-USD.csv', 'r', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(reader)
    for row in reader:
        count += 1
        newrow = row[0].split(',')
        if prev is None:
            prev = float(newrow[4])
            continue
        diff = float(newrow[4])-prev
        if abs(diff/prev) < 0.05:
            continue
        sign = "1"
        if diff < 0:
            sign = "0"
        history[newrow[0]] = sign
        prev = float(newrow[4])
print(count, len(history))
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
