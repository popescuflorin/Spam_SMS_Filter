import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv('dataset\SMSSpamCollection', header=None, sep='\t', names=['Label', 'SMS'])

df=df.rename({'Label':'target',
             'SMS':'text'},axis=1)

len_text=[]
for i in df['text']:
    len_text.append(len(i))

df['text_length']=len_text

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df['target']=np.where(df['target']=='spam',1,0)


spam=[]
ham=[]
spam_class=df[df['target']==1]['text']
ham_class=df[df['target']==0]['text']

def extract_ham(ham_class):
    global ham
    words = [word.lower() for word in word_tokenize(ham_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    ham=ham+words

def extract_spam(spam_class):
    global spam
    words = [word.lower() for word in word_tokenize(spam_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    spam=spam+words

spam_class.apply(extract_spam)
ham_class.apply(extract_ham )

print(spam_class)
print(ham_class)

spam_words=np.array(spam)
pd.Series(spam_words).value_counts().head(n=10)

ham_words=np.array(ham)
pd.Series(ham_words).value_counts().head(n=10)

from nltk.stem import SnowballStemmer
import string
stemmer = SnowballStemmer("english")

def cleanText(message):
    
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    
    return " ".join(words)

df["text"] = df["text"].apply(cleanText)
print(df.head(n = 10)) 

y=df['target']
x=df['text']

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(x_train)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
joblib.dump(cv, 'cv.pkl')
print(type(x_train))

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_train,y_train)
pred_2=classifier.predict(cv.transform(x_test))
score_2=accuracy_score(y_test,pred_2)
print(score_2)
print("_______________")




# Save your model

joblib.dump(classifier, 'model.pkl')
print("Model dumped!")


# Load the model that you just saved
classifier = joblib.load('model.pkl')
# Saving the data columns from training
