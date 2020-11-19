import pandas as pd
import numpy as np
import matplotlib as mp
import json
#from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
# TfidfVectorizer is used for checking terms
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import random
from sklearn.model_selection import GridSearchCV
import pickle






class Sentiment:
    NEGATIVE ="NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
    def __init__(self,text,score,time):
        self.text = text
        self.score = score
        self.time = time
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <=2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


class ReviewContainer:
    def __init__(self,reviews):
        self.reviews= reviews

    def evenly_distribute(self):
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))


        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

filename = 'Books_small_10000.json'
reviews =[]
with open(filename) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall'],review['reviewTime']))

#print(reviews[5].sentiment)
#print(len(reviews))




# PREPARE DATA TRAINING TESTING


# spliting our data for training and testing
training,test=train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)





#print("Length of training data : ",len(training))
#print("Length of testing data : ",len(test))
#print(training[0].sentiment)

train_container.evenly_distribute()

train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

train_y.count(Sentiment.POSITIVE)
train_y.count(Sentiment.NEGATIVE)

# Bag of words vectorization

vectorizer = CountVectorizer()

# it will fit and transform your model
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
#print(train_x[0])
#print(train_x_vector[0])






# Classification google it
# Linear SVM

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors,train_y)

clf_svm.fit(train_x_vectors, train_y)

#print(test_x[0])

#print(clf_svm.predict(test_x_vectors[90]))

# DECISION TREE


clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors,train_y)

clf_dec.predict(test_x_vectors[0])

# GAUSSIAN NAIVE BAYES,

clf_gnb =DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors,train_y)

clf_gnb.predict(test_x_vectors[0])

# LOGISTIC REGRESSION


clf_log = LogisticRegression()
clf_log.fit(train_x_vectors,train_y)

clf_log.predict(test_x_vectors[0])

# EVALUATION every model
# Mean accuracy
'''
print(clf_svm.score(test_x_vectors,test_y))
print(clf_dec.score(test_x_vectors,test_y))
print(clf_gnb.score(test_x_vectors,test_y))
print(clf_log.score(test_x_vectors,test_y),"\n")
'''
# F1 SCORES
#'''
print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print(f1_score(train_y, clf_svm.predict(train_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
#print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[ Sentiment.POSITIVE, Sentiment.NEGATIVE]))
#print(f1_score(test_y, clf_gnb.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
#print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
#'''

#var= input()
#test_set = [var]
test_set=['I thouroughly enjoy this, 5 star',"bad look do not but", 'horrible waste of time','I love this book']
new_test = vectorizer.transform(test_set)
print(clf_svm.predict(new_test))




# Improving our model

'''
print(train_y.count(Sentiment.NEGATIVE))
print(train_y.count(Sentiment.POSITIVE))
print(test_y.count(Sentiment.POSITIVE))
print(test_y.count(Sentiment.NEGATIVE))
'''

# Tunning our model (with grid search)

parameters = {'kernel':('linear','rbf'), 'C':(1,4,8,16,32)}


svc=svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
print(clf.fit(train_x_vectors,train_y))

# More improvement
# Model saving


with open('./sentiment_classifer.pk1','wb') as f:
    pickle.dump(clf,f)

# Load model


with open('./sentiment_classifer.pk1','rb') as f:
    loaded_clf = pickle.load(f)

print(test_x[0])
print(loaded_clf.predict(test_x_vectors[0]))



































