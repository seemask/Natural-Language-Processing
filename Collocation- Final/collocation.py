# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:27:17 2020

@author: Seema S Kanaje
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams

import string
import re
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from collections import Counter
from nltk.probability import FreqDist


def get_clean_token_list(words):
    clean_words=[]
    for word in words:
        flag=True
        for letter in word:
            if letter.isdigit() or letter in string.punctuation:
                flag=False
        if flag:
            clean_words.append(word)
    return clean_words
def remove_stopwords(words):
    """
    pass series get series
    """
    filtered_sent=[]
    for word in words:
        if word not in stop_words:
            filtered_sent.append(word)
    return filtered_sent
def lemmatize_sentence(sentence):
    lem_sentence=[]
    for word in sentence:
        lem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    return lem_sentence
def chisquaretest():
    list_bigrams=[]
    total_bigrams = bigram_df['frquency'].sum() 
    total_unigrams=unigram_df['frquency'].sum() 
    for (index,row) in bigram_df.iterrows():
        
        leftword=row.token.split()[0]
        rightword=row.token.split()[1]
        
        
        O11=float(bigram_df[bigram_df["token"]==row.token].frquency)
        O12=float((unigram_df[unigram_df["token"]==rightword].frquency))-float(O11)
        O21=float((unigram_df[unigram_df["token"]==leftword].frquency))-float(O11)
        O22=float(total_bigrams-O11)
        
        
        chisquare=float((((O11*O22)-(O12*O21))**2)*total_bigrams)/float((O11+O12)*(O11+O21)*(O12+O22)*(O21+O22))

        list_bigrams.append([row.token,chisquare])
       
    return list_bigrams
def pontwise_mutual_info():
    PMI_bigrams=[]
    total_bigrams = bigram_df['frquency'].sum() #217491
    total_unigrams=unigram_df['frquency'].sum() #230383
    for (index,row) in bigram_df.iterrows():
        leftword=row.token.split()[0]
        rightword=row.token.split()[1]
        C_W1_W2=float(bigram_df[bigram_df["token"]==row.token].frquency)
        C_W1=float(unigram_df[unigram_df["token"]==leftword].frquency)
        C_W2=float(unigram_df[unigram_df["token"]==rightword].frquency)
        P_W1_W2=C_W1_W2/total_bigrams
        P_W1=C_W1/total_unigrams
        P_W2=C_W2/total_unigrams
        PMI=math.log(P_W1_W2/float(P_W1*P_W2),2)
        PMI_bigrams.append([row.token,PMI])
    return PMI_bigrams



df = pd.read_csv(r'D:\Fall2020\NLP\Assignment 1\Collocations',sep="\n", header=None)
stop_words=set(stopwords.words("english"))

df.columns=['text']

df['text']=df['text'].str.lower()

df['tokens']=df['text'].apply(lambda x: word_tokenize(x))

df['tokens']=df['tokens'].apply(lambda words: [word.lower() for word in words if word.isalpha()])

df['tokens']=df['tokens'].apply(remove_stopwords)

wordnet_lemmatizer = WordNetLemmatizer()
df['lemm_sentence']=df['tokens'].apply(lambda x: lemmatize_sentence(x))
df['clean_sentence']=df['lemm_sentence'].apply(lambda x: ' '.join(x))

reviews=[word for review in df['lemm_sentence'] for word in review]
fdist_reviews = FreqDist(reviews)


vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
dataset_vector = vectorizer.fit_transform(df["clean_sentence"])
list_tokens=[]
for key, val in vectorizer.vocabulary_.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if key in fdist_reviews:
            list_tokens.append([key,val,fdist_reviews[key]])
unigram_df=pd.DataFrame(list_tokens,columns=['token','index','frquency'])     


s=' '.join(list(df['clean_sentence']))

s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
tokens = [token for token in s.split(" ") if token != ""]

# 2grams
bgs = nltk.bigrams(tokens)

#compute frequency distribution for all the bigrams in the text
bigram_fdist = nltk.FreqDist(bgs)
tmp_bigram_frequency_tokens={ x[0]:x[1] for x in bigram_fdist.most_common()}
# pp.pprint(bigram_frequency_tokens)   


bigram_frequency_tokens={}
for key , value in tmp_bigram_frequency_tokens.items():
    bigram_frequency_tokens[key[0]+" "+key[1]]=value
    
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', ngram_range=(2, 2))
dataset_vector = vectorizer.fit_transform(df["clean_sentence"])


# chi2_selector = chi2(dataset_vector, X_train['label'])

list_tokens=[]
for key, val in vectorizer.vocabulary_.items():
        if key in bigram_frequency_tokens:
            list_tokens.append([key,val,bigram_frequency_tokens[key]])    
bigram_df=pd.DataFrame(list_tokens,columns=['token','index','frquency'])            

print('Chi Square Test results')
chi_bigrams=chisquaretest()
chi2_df=pd.DataFrame(chi_bigrams,columns=["Bigrams", "Chi2"])
print(chi2_df.sort_values(by='Chi2',ascending=False).head(20))
print('Pointwise Mutual Information results')
PMI=pontwise_mutual_info()
PMI_df=pd.DataFrame(PMI,columns=["Bigrams", "PMI"])
print(PMI_df.sort_values(by='PMI',ascending=False).head(20))