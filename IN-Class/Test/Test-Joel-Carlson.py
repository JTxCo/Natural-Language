import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

"""
Reads csv file into dataframe.
Gets category labels and documents text.
"""
df = shuffle(pd.read_csv("website_classification.csv"))
labels = df["Category"]
print(df["Category"].value_counts())
documents = df["cleaned_website_text"].dropna()

"""
Labelencoding.
"""
le = LabelEncoder()
y = le.fit_transform(labels)

"""
Initialized vectorizers and naive bayes.
"""
cv = CountVectorizer()
tfidf = TfidfVectorizer()
nb = MultinomialNB()

"""
Initalized stemmers and lemmatizers.
The lists are for you to store the stemmed and lemmatized results.
"""
stemmer = PorterStemmer()
porter = []
stemmer2 = SnowballStemmer("english")
snowball = []
lemmatizer = WordNetLemmatizer()
wordnet = []

for doc in documents:
    """
    TODO:
    Apply different stemming and lemmatization methods.
    Add adjusted documents to the relevant lists.
    """
    token_words = nltk.word_tokenize(doc)
    lemma_sentence = ' '.join(lemmatizer.lemmatize(word) for word in token_words)
    wordnet.append(lemma_sentence)
    snowball_stem_sentence = ' '.join(stemmer2.stem(word) for word in token_words)
    snowball.append(snowball_stem_sentence)
    porter_stem_sentence = ' '.join(stemmer.stem(word) for word in token_words)
    porter.append(porter_stem_sentence)
    

datasets = [documents, porter, snowball, wordnet]
dataset_names = ["regular", "porter", "snowball", "lemmatized"]
for i, data in enumerate(datasets):
    print("\n\n", dataset_names[i])
    kfold = KFold(n_splits=5)
    """
    TODO:
    Vectorize the data using BoW and TFIDF
    """
    countVectorizer = CountVectorizer()
    tfidfVectorizer = TfidfVectorizer()
    bow_X = countVectorizer.fit_transform(data)
    tfidf_X = tfidfVectorizer.fit_transform(data)
    
    vector_data = [bow_X, tfidf_X]
    vector_names = ["BoW", "TFIDF"]
    
    for j, X in enumerate(vector_data):
        acc = []
        f1_micro = []
        f1_macro = []
        # print(vector_names[j])
        for (train_index, test_index) in kfold.split(documents):
            """
            TODO:
            Fit the naive bayes model on both BoW and TfIDF transformed data.
            """
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            acc.append(accuracy_score(y_test, y_pred))
            f1_micro.append(f1_score(y_test, y_pred, average="micro"))
            f1_macro.append(f1_score(y_test, y_pred, average="macro"))
        acc = np.array(acc)
        f1_micro = np.array(f1_micro)
        f1_macro = np.array(f1_macro)
        
        print("dataset: ", dataset_names[i])
        print("vectorizer: ", vector_names[j])
        print("Accuracy: %0.4f (+/- %0.4f)" % (acc.mean(), acc.std() * 2))
        print("F1-micro: %0.4f (+/- %0.4f)" % (f1_micro.mean(), f1_micro.std() * 2))
        print("F1-macro: %0.4f (+/- %0.4f)" % (f1_macro.mean(), f1_macro.std() * 2))
        print("\n")
        """
        TODO:
        Get the average accuracy, f1-micro and f1-macro score for the different combinations
        """
        
