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
nltk.download('wordnet')
nltk.download('punkt')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import locale
locale.getpreferredencoding = lambda: "UTF-8"
import en_core_web_md
text_to_nlp = spacy.load('en_core_web_md')


import wordcloud
from wordcloud import WordCloud





def review_is_good(number_of_stars):
    if number_of_stars>3:
        return True
    else:
        return False

def tokenize(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): # -PRON- is a special all inclusive "lemma" spaCy uses for any pronoun, we want to exclude these
            clean_tokens.append(token.lemma_)
    return clean_tokens

df = pd.read_csv("yelp_dataset.csv")

df = df[["stars", "text"]]

#Generating a Word Cloud based on number of stars
#stars = 5
#text = ""
#for text in df[df['stars'] == stars]['text'].values:
#   text += text + ' '
#wordcloud = WordCloud()
#wordcloud.generate_from_text(text)
#plt.figure(figsize=(14,7))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.show()

#New column for good/bad reviews
df["is_good_review"] = df["stars"].apply(review_is_good)


X_text = df['text']
y = df['is_good_review']


#Tokenizing & Converting to Bag of Words
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X_text)
X = bow_transformer.transform(X_text)
pd.DataFrame(X.toarray())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


#Creating model with Sklearn
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_predictions = logistic_model.predict(X_test)
accuracy = accuracy_score(y_predictions, y_test)
print("Logistic Regression accuracy:", accuracy)


#Using Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train,y_train)

y_predictions = nb_model.predict(X_test)
accuracy = accuracy_score(y_predictions,y_test)
print("Naive Bayes accuracy:", accuracy)