#import sys
import sys
#import json
import json
#import panda
import pandas as pd

#import random
import random
#import SnowballStemmer and Stopwords from nltk library
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#import words
from nltk.corpus import words
# Import Pipeline
from sklearn.pipeline import Pipeline

import re
#imoport scikit learn library
import sklearn
#Import  TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import Shuffle
from sklearn.utils import shuffle


# Import Multinominal Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB

#Import train_test_split to split dataset into training and testing sets
from sklearn.model_selection import train_test_split



with open('C:\Classifier\IT.json', 'r') as file:
    jsonfile=json.load(file)

data = pd.DataFrame(jsonfile) #IT


with open('C:\Classifier\operation.json', 'r') as file:
    jsonfile2=json.load(file)

data2 = pd.DataFrame(jsonfile2) #OPERATIONS

data = data.append(data2)


with open('C:\Classifier\Other.json', 'r') as file:
    jsonfile3=json.load(file)

dataOther = pd.DataFrame(jsonfile3) #OTHER

data = data.append(dataOther)


#function for stemming
def Stemmer(i):
    Stemmer = SnowballStemmer("english")
    return Stemmer.stem(i)



def nbclassifier(data,email):
    #takes away stop words
    stopwd = stopwords.words("english")


    #re.sub("[^a-zA-Z], " ",x) #strip all characters and removes punctions and any character that is not an alphabet

    #json format for text to analyze
    text = [{"message":""}]
    #assign message to text to analyzr
    text[0]['message'] = email

    #convert it to data to analyze
    data3 = pd.DataFrame(text)
    

    #apply() applies action to all values in dataFrame
    #lambda combines  multiple arguments as one
    #lower() takes every string to lowercase
    data['processed'] = data['Message'].apply(lambda x: " ".join([Stemmer(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwd]).lower())

    data3['new'] = data3['message'].apply(lambda x: " ".join([Stemmer(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwd]).lower())


    stemedtext = data3['new'][0]
    

    

    X_train, X_test, y_train, y_test = train_test_split(data['processed'],data.Class, test_size=0.1)

    #TfidfVectorizer creates a matrix of bag of words, gives words a weight
    #ngram_range specifies which ngram size to use set to one
    #stop_words stops can get very large in corpus hence increase model size hence stop_words can be used to deal with that
    #sublinear_tf = True uses a logarthmic function instead of a natural # its much better
    # ---->TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)



    #penalty specifies whether to use l1 or l2 penalization
    # ----->LinearSVC(C=1.0,penalty="l1",dual=False)
    # -----> LinearSVC(C=1.0,penalty="l2",dual=True)
    # -----> LogisticRegression(solver='liblinear')
    #----->MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    # ---> KMeans(n_clusters=2, max_iter=100, n_init=1)



    process = [("Vectorizer",TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)),
               ("clf",MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))]
    #
    pipe = Pipeline(process)

    

    model = pipe.fit(X_train, y_train)



    #print("accuracy score: " + str(model.score(X_test, y_test)))


    val = model.predict([stemedtext])

    

    if val == ['0']:
        #f.write("IT")
        print("IT")
        #f.close()

    elif val == ['1']:
        #f.write("Operations")
        print("Operations")
        #f.close()

    elif val == ['2']:
        #f.write("Other")
        print("Other")
        #f.close()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("add email argument")
        
    elif sys.argv[1] == "":
        print("add an email")    
    else:
        nbclassifier(data,sys.argv[1])
        
