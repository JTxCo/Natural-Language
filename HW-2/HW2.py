import pandas as pd
import nltk

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

"""
Seperates out the data to train on (X) and the labels we wish to classify (y).
"""
spam_df = pd.read_csv("SPAM.csv")
spam_y = spam_df["Category"] # what we are trying to classify!
spam_X = spam_df["Message"] # What we are training our model on -> convert to BoW vector

"""
Perform vectorization and label encoding
"""


"""
Create a 5-fold cross validation experiment using Naive Bayes.
Calculate the average precision, accuracy, recall, and f1-scores across the five folds.
"""

