__author__ = 'Michael'

import pandas as pd
import numpy as np
import string

from collections import Counter
base_file = 'for_matching2.csv'


dat = pd.read_csv(base_file)


import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


print 'lets go!'


dat['slicer'] = 0

#takes annoyingly long zzz 200K rows
for index, row in dat.iterrows():
    text2 = row['text'][:2]

    if text2 == 'RT':
        dat.loc[index,'slicer'] = 1
    #dat.loc[index,'text'] = remove_htmls(text)
dat = dat[dat['slicer']!= 1]

dat['text'] = dat['text'].str.lower()

dat = dat.drop_duplicates(subset='id', keep= 'last')


#build the dictionary

token_dict = {}

#a few have very few samples sub 800
drop_list = ['joniernst','SaxbyChambliss','SenatorRisch',
             'SenatorRisch','POTUS','d', 'senatorsanders',
             'repcorygardner', 'RepTomCotton','SenBillNelson']

#325 is the lowest
#hacking it for first row

name_prev= 'EmmaWatson'
i = 1
dat['num_per_user'] = 0

for index, row in dat.iterrows():
    name = row['name']
    if name != name_prev:
        i = 1
        dat.loc[index, 'num_per_user'] = 1
    if name == name_prev:
        if i <= 900:
            dat.loc[index,'num_per_user']= 1
            i+=1
        if i > 900 and i <1000:
            dat.loc[index,'num_per_user']= 2

            continue

    name_prev = row['name']

dat['num_per_user'].value_counts()

for i in drop_list:
    dat = dat[dat['name'] != i]

test = dat[dat['num_per_user'] == 2]
train = dat[dat['num_per_user'] == 1]


#### combine text into a large body per user
name_prev= 'EmmaWatson'
full_text = ''

test_dict = {}
train_dict = {}
for index, row in train.iterrows():
    name = row['name']
    if name != name_prev:
        train_dict[name_prev] = full_text
        full_text = ""
    if name == name_prev:
        full_text = full_text + '   ' + row['text']

    name_prev = row['name']

#train = pd.DataFrame(train_dict.items(), columns=['name', 'test'])
print 'tokens'

name_prev= 'EmmaWatson'
full_text = ''


for index, row in test.iterrows():
    name = row['name']
    if name != name_prev:
        test_dict[name_prev] = full_text
        full_text = ""
    if name == name_prev:
        full_text = full_text + '   ' + row['text']

    name_prev = row['name']

#test = pd.DataFrame(test_dict.items(), columns=['name', 'test'])

print 'tfidf'
#make character level bigrams
tfidf = TfidfVectorizer(analyzer='char',lowercase=False,ngram_range = (2,2),max_df = .8, min_df= .10)

tfs = tfidf.fit_transform(train_dict.values())

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


clf = svm.SVC(kernel='linear', C=1,decision_function_shape =  'ovr' )
nb = MultinomialNB().fit(tfs, train_dict.keys())
print 'scoring???'
model_svm = clf.fit(tfs,train_dict.keys())
linsvc = LinearSVC().fit(tfs, train_dict.keys())

test_tfs = tfidf.transform(test_dict.values())


predictions_svc = model_svm.predict(test_tfs)
predictions_nb = nb.predict(test_tfs)
predictions_lsvc = linsvc.predict(test_tfs)

# 58
np.mean(predictions_svc == test_dict.keys())
#63
np.mean(predictions_nb == test_dict.keys())
#62
np.mean(predictions_lsvc == test_dict.keys())





