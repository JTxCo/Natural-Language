import math

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import reuters

documents = [reuters.raw(doc) for doc in reuters.fileids()]
categories = [reuters.categories(doc)[0] for doc in reuters.fileids()]

"""
1.
Implement CountVectorizer from Scikit-learn
to transform documents into Bag of Words representation
"""

CV = CountVectorizer()
docs = CV.fit_transform(documents)
print(docs.shape)



"""
2.
Implement LabelEncoder from Scikit-learn
to encode class labels (categories)
"""
le = LabelEncoder()
categories = le.fit_transform(categories)

print(categories.shape)
"""
3.
Split off part of the data for training and testing (80/20 split)
"""
train_docs, test_docs, train_cats, test_cats = train_test_split(docs, categories, test_size=0.2, random_state=42)


"""
4.
Apply Naive Bayes to train the model on training data (80%)
"""

classifier = MultinomialNB()
classifier.fit(train_docs, train_cats)



"""
5.
Predict, using trained Naive Bayes model, what class unseen articles (testing data) belong to
"""
train_cats_pred = classifier.predict(train_docs)

"""
6.
Calculate accuracy for these predictions
"""

accuracy_score = accuracy_score(train_cats, train_cats_pred)
print(accuracy_score)

