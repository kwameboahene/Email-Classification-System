#import sys
import sys
#import json
import json
#import panda
import pandas as pd
import numpy as np
#import random
import random
#import SnowballStemmer and Stopwords from nltk library
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#import words
from nltk.corpus import words

import matplotlib.pyplot as plt
import re
#imoport scikit learn library
import sklearn
#Import  TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import Shuffle
from sklearn.utils import shuffle
# Import SVC
from sklearn.svm import LinearSVC
# Import Pipeline
from sklearn.pipeline import Pipeline

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import precision_recall_curve

  

# Import Multinominal Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB

#Import train_test_split to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

#import logisticRegression Modal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

with open('C:\Classifier\IT.json', 'r') as file:
    jsonfile=json.load(file)

data = pd.DataFrame(jsonfile) #IT


with open('C:\Classifier\operation.json', 'r') as file:
    jsonfile2=json.load(file)

data2 = pd.DataFrame(jsonfile2) #OPERATIONS

data = data.append(data2)


with open('C:\Classifier\Other.json', 'r') as file:
    jsonfile3=json.load(file)

dataOther = pd.DataFrame(jsonfile3) #Other

data = data.append(dataOther)




def nbclassifier(data,newtext):
    #takes away stop words
    stopwd = stopwords.words("english")

    #takes all forms of a word and treats it as one eg(coming,came,comes == come)
    Stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer() 


    #re.sub("[^a-zA-Z], " ",x) #strip all characters and removes punctions and any character that is not an alphabet

    #json format for text to analyze
    text = [{"message":""}]
    #assign message to text to analyzr
    text[0]['message'] = newtext

    #convert it to data to analyze
    data3 = pd.DataFrame(text)
    
    
    #apply() applies action to all values in dataFrame
    #lambda combines  multiple arguments as one
    #lower() takes every string to lowercase
    data['processed'] = data['Message'].apply(lambda x: " ".join([Stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwd]).lower())


    data3['new'] = data3['message'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwd]).lower())


    stemedtext = data3['new'][0]

    print(stemedtext)

    

    X_train, X_test, y_train, y_test = train_test_split(data['processed'],data.Class, test_size=0.2)

    #TfidfVectorizer creates a matrix of bag of words, gives words a weight
    #ngram_range specifies which ngram size to use set to one
    #stop_words stops can get very large in corpus hence increase model size hence stop_words can be used to deal with that
    #sublinear_tf = True uses a logarthmic function instead of a natural # its much better
    # ---->TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)


##    testray = ["The internet at my hostel is not working", "Dear Support, I would like to report the beds in my hostel have a problem"]
##    print(testray)
##    Vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words="english")
##    print(Vectorizer.fit_transform(testray).toarray())

    

    process = [("Vectorizer",TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)),
               ("clf",MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))]
    
    
    pipe = Pipeline(process)

    rayy = []
    ray3 = []
    ray3 = y_test.values
    for i in ray3:
        if i == '0':
            t = 0
            rayy.append(t)
        elif i == '1':
            t = 1
            rayy.append(t)
        if i == '2':
            t = 2
            rayy.append(t)

    print(rayy)

    ray1 = []
    ray1 = X_test.values
    ray =[]
    for i in ray1:
        ray.append(i)

    model = pipe.fit(X_train, y_train)

    Ray4=[]

    for i in ray:
        val= model.predict([i])
        if val == ['0']:
            t = 0
            Ray4.append(t)
        elif val == ['1']:
            t = 1
            Ray4.append(t)
        if val == ['2']:
            t = 2
            Ray4.append(t)

    target_names = ['IT', 'Operations', 'Other']

    print(classification_report(rayy, Ray4, target_names=target_names))
    #print(X_test)

    #print(y_test)
    
  

    






    f = open("new.txt","w")
    f.write(stemedtext)
    f.close()


    print("accuracy score: " + str(model.score(X_test, y_test)))


    val = model.predict([stemedtext])
    val = model.predict([stemedtext])
    val = model.predict([stemedtext])
    

    if val == ['0']:
        #f.write("IT")
       # print("IT")
        f.close()

    elif val == ['1']:
        #f.write("Operations")
        #print("Operations")
        f.close()

    elif val == ['2']:
        #f.write("Other")
     #   print("Other")
        f.close()


nbclassifier(data,"Dear Support, the lights in my hostel do not work")
