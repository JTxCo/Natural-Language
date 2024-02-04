import pandas as pd
import nltk

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

"""
Seperates out the data to train on (X) and the labels we wish to classify (y).
"""
spam_df = pd.read_csv("HW2_SPAM.csv")
spam_y = spam_df["Category"] # what we are trying to classify!
spam_X = spam_df["Message"] # What we are training our model on -> convert to BoW vector

"""
Perform vectorization and label encoding
"""

countVectorizer = CountVectorizer()
X = countVectorizer.fit_transform(spam_X)
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(spam_y)


# print(labelEncoder.classes_)


"""
Create a 5-fold cross validation experiment using Naive Bayes.
Calculate the average precision, accuracy, recall, and f1-scores across the five folds.
"""

KF = KFold(n_splits=5, shuffle=True, random_state=42)
# KF.get_n_splits(X)

# accuracy scores
classifier = MultinomialNB()
average_percision_scores = cross_val_score(classifier, X, y, cv=KF, scoring='precision_macro')
print("Precision: %0.4f (+/- %0.4f)" % (average_percision_scores.mean(), average_percision_scores.std() * 2))

accuracy_scores = cross_val_score(classifier, X, y, cv=KF, scoring='accuracy')
print("Accuracy: %0.4f (+/- %0.4f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))

# recall scores
recall_scores = cross_val_score(classifier, X, y, cv=KF, scoring='recall_macro')
print("Recall: %0.4f (+/- %0.4f)" % (recall_scores.mean(), recall_scores.std() * 2))

# f1 scores
f1_scores = cross_val_score(classifier, X, y, cv=KF, scoring='f1_macro')
print("F1: %0.4f (+/- %0.4f)" % (f1_scores.mean(), f1_scores.std() * 2))


