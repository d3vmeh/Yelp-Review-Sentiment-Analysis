import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import wordcloud


nltk.download('wordnet')
nltk.download('punkt')

from wordcloud import WordCloud


df = pd.read_csv("yelp_dataset.csv")

df = df[["stars", "text"]]
print(df.head())

stars = 5
text = ""
for text in df[df['stars'] == stars]['text'].values:
    text += text + ' '

wordcloud = WordCloud()
wordcloud.generate_from_text(text)
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
print("plot shown")