from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
import numpy as np

documents = [reuters.raw(doc) for doc in reuters.fileids()]
categories = [reuters.categories(doc)[0] for doc in reuters.fileids()]

"""
1.
encode class labels (categories)
"""
LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform(categories)



"""
2. Use the snowball stemmer
What options are there?
How many languages does the snowball stemmer support? 15
"""






stop_words = set(stopwords.words('english'))

stemer = SnowballStemmer("english")
stemmed_docs = []
for doc in documents:
    token_words = word_tokenize(doc)
    stem_sentence = ' '.join(stemer.stem(word) for word in token_words if word.lower() not in stop_words)
    stemmed_docs.append(stem_sentence)




"""
3. Apply both BoW and TF-IDF on the stemmed documents and on the original documents
"""


# BOW
countVectorizer = CountVectorizer()
Original_X_BOW = countVectorizer.fit_transform(documents)
Stemmed_X_BOW = countVectorizer.fit_transform(stemmed_docs)


# TF-IDF
tfidf = TfidfVectorizer()
Original_X_tfidf = tfidf.fit_transform(documents)
Stemmer_X_tfidf = tfidf.fit_transform(stemmed_docs)





"""
4. Using 5-fold CV,
train NB using the four different vectorized documents
and evaluate using macro F1 score and accuracy.
Think about how to do this efficiently so you dont write 4 separate cv experiments
"""


# Original_X_BOW

KF = KFold(n_splits=5, shuffle=True, random_state=42)

classifier = MultinomialNB()

datasets = [Original_X_BOW, Stemmed_X_BOW, Original_X_tfidf, Stemmer_X_tfidf]

for i, X in enumerate(datasets):
  
  # Store accuracy and f1 scores for each fold
  accuracies = []
  f1_scores = []
  
  for train_index, test_index in KF.split(X):
    #   setting up train and test sets to make it easier to read. can work without doing it this way
      train_docs, test_docs, train_cats, test_cats = X[train_index], X[test_index], y[train_index], y[test_index]
      classifier.fit(train_docs, train_cats)
      test_cats_pred = classifier.predict(test_docs)
      
      accuracy = accuracy_score(test_cats, test_cats_pred)
      f1 = f1_score(test_cats, test_cats_pred, average='macro')
      
      accuracies.append(accuracy)
      f1_scores.append(f1)

  # Convert to numpy arrays for mean and std calculation
  accuracies = np.array(accuracies)
  f1_scores = np.array(f1_scores)

  # Dataset names
  names = ['Original_X_BOW', 'Stemmed_X_BOW', 'Original_X_tfidf', 'Stemmer_X_tfidf']

  print('\nDataset: ', names[i])
  print("Accuracy: %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std() * 2))
  print("F1: %0.4f (+/- %0.4f)" % (f1_scores.mean(), f1_scores.std() * 2))

