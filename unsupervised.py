#import json
import json
#import panda
import pandas as pd
#import SnowballStemmer and Stopwords from nltk library
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#import words
from nltk.corpus import words


import re
#imoport scikit learn library
import sklearn
#Import  TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Import SVC
from sklearn.svm import LinearSVC
# Import Pipeline
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans


with open('inbox.json', 'r') as file:
    jsonfile=json.load(file)

data = pd.DataFrame(jsonfile)

#takes away stop words
stopwd = stopwords.words("english")

#takes all forms of a word and treats it as one eg(coming,came,comes == come)
Stemmer = SnowballStemmer("english")

#apply() applies action to all values in dataFrame
#lambda combines  multiple arguments as one
#lower() takes every string to lowercase
data['processed'] = data['Message'].apply(lambda x: " ".join([Stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwd]).lower())



#TfidfVectorizer creates a matrix of bag of words, gives words a weight
#ngram_range specifies which ngram size to use set to one
#stop_words stops can get very large in corpus hence increase model size hence stop_words can be used to deal with that
#sublinear_tf = True uses a logarthmic function instead of a natural # its much better
# ---->TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)




Vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words="english",sublinear_tf=True)

Transformed = Vectorizer.fit_transform(data['processed'])

Model= KMeans(n_clusters=2, max_iter=100)

Model.fit(Transformed)

#get the feature names for each cluster
featuresname = Vectorizer.get_feature_names()
order = Model.cluster_centers_.argsort()[:, ::-1]

           
for i in range(2):
    print("Cluster %d:" % i),
    for features in order[i, :10]:
        print(" %s" % featuresname[features])


test = ["Good Evening I would to know when the next bus leaves campus Regards, Kwame Boahene"]
Transform = Vectorizer.transform(test)

print(Model.predict(Transform))

        
        


