#Sentiment Analysis
#By Gabrieli Silva
import nltk
import re
import pandas as pd
import tweepy
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

ds = pd.read_csv('tweets.csv')
df = pd.DataFrame(ds)

contador = 0
resultado_total = []

new_result = []
new_pol = []
new_sub = []

cont_neutro = 0
cont_posit = 0
cont_negat = 0

#In this code assume that we have 200 tweets from an account to analyze
while (contador < 200):
    tweets = df["tweets"][contador]

    frase = tb(tweets)

    print(frase.tags)
    
    polarity=frase.polarity
    print("Polarity:",polarity)
    
    subjectivity=frase.subjectivity
    print("Subjectivity:",subjectivity)
    
    resultados = [polarity,subjectivity]
    
    resultado_total.append(resultados)
    
    if(polarity>0.0):
        pol='positivo'
        cont_posit = cont_posit +1
    else:
        if(polarity<0.0):
            pol = 'negativo'
            cont_negat = cont_negat +1
        else:
            pol='neutro'
            cont_neutro = cont_neutro +1
    
    result = [pol,subjectivity]
    
    new_result.append(result)
    new_pol.append(pol)
    new_sub.append(subjectivity)
    
    contador = contador + 1

#Drawing the pie chart with the distribution of feelings.
labels = ['Positivo', 'Negativo', 'Neutro']
sizes = [cont_posit, cont_negat, cont_neutro]
fig1,ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
ax1.axis('equal')
   
plt.show()
        
        
