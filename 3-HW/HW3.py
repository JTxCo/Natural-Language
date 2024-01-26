"""
These are all the imports you should need.
"""
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

"""
Reads csv file into dataframe.
Gets category variables.
Combines the two relevant features after stripping unnecessary characters.
"""
df = pd.read_csv("HW3_reduced_huffpost.csv", encoding="latin-1") #encoding ensures special characters are being read in correctly
labels = df["category"]
df["headline"] = df["headline"].str.strip("b\'\"")
df["short_description"] = df["short_description"].str.strip("b\'\"")
documents = df["headline"] + df["short_description"]

"""
Perform labelencoding.
"""

LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform(labels)


"""
Apply the different types of stemming and lemmatization.
"""
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

# stemming and lemmitization combiend here but commented out separated below
snowball_stemer = SnowballStemmer("english")
snowball_stemmed_docs = []
porter_stemer = PorterStemmer()
porter_stemmed_docs = []
lemmatizer = WordNetLemmatizer()
lemmatized_docs = []

for doc in documents:
    token_words = word_tokenize(doc)
    lemma_sentence = ' '.join(lemmatizer.lemmatize(word) for word in token_words if word.lower() not in stop_words)
    lemmatized_docs.append(lemma_sentence)   
    snowball_stem_sentence = ' '.join(snowball_stemer.stem(word) for word in token_words if word.lower() not in stop_words)
    snowball_stemmed_docs.append(snowball_stem_sentence)
    porter_stem_sentence = ' '.join(porter_stemer.stem(word) for word in token_words if word.lower() not in stop_words)
    porter_stemmed_docs.append(porter_stem_sentence)
    
    
# lemmatization
# lemmatizer = WordNetLemmatizer()
# lemmatized_docs = []
# for doc in documents:
#     token_words = word_tokenize(doc)
#     lemma_sentence = ' '.join(lemmatizer.lemmatize(word) for word in token_words if word.lower() not in stop_words)
#     lemmatized_docs.append(lemma_sentence)


"""
Send the results through the different vectorizer models.
BoW and TF-IDF are directly available in sci-kit learn.
As we saw in class, we need to create a pipeline for LSA using SVD and MinMaxScaler.
"""

# Each is done on none, porter, snowball, and lemmatized

# BOW vectorizer
CountVectorizer =  CountVectorizer()
none_X_BOW = CountVectorizer.fit_transform(documents)
porter_X_BOW = CountVectorizer.fit_transform(porter_stemmed_docs)
snowball_X_BOW = CountVectorizer.fit_transform(snowball_stemmed_docs)
lemmatized_X_BOW = CountVectorizer.fit_transform(lemmatized_docs)


# TF-IDF vectorizer
TfidfVectorizer = TfidfVectorizer()
none_X_tfidf = TfidfVectorizer.fit_transform(documents)
porter_X_tfidf = TfidfVectorizer.fit_transform(porter_stemmed_docs)
snowball_X_tfidf = TfidfVectorizer.fit_transform(snowball_stemmed_docs)
lemmatized_X_tfidf = TfidfVectorizer.fit_transform(lemmatized_docs)


# LSA
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, MinMaxScaler())
none_X_lsa = lsa.fit_transform(none_X_tfidf)
porter_X_lsa = lsa.fit_transform(porter_X_tfidf)
snowball_X_lsa = lsa.fit_transform(snowball_X_tfidf)
lemmatized_X_lsa = lsa.fit_transform(lemmatized_X_tfidf)











"""
Using 10-fold cross validation evaluate the different vector representations.
Train and test using the Naive Bayes model.
Calculate the average accuracy and (micro and macro) F1-scores.
"""



KF = KFold(n_splits=10, shuffle=True, random_state=42)

classifier = MultinomialNB()

# all of the datasets created above to loop through
datasets = [none_X_BOW, porter_X_BOW, snowball_X_BOW, lemmatized_X_BOW, none_X_tfidf, porter_X_tfidf, snowball_X_tfidf, lemmatized_X_tfidf, none_X_lsa, porter_X_lsa, snowball_X_lsa, lemmatized_X_lsa]

for i, x in enumerate(datasets):
    accuracies = []
    f1_scores_macro = []
    f1_scores_micro = []
    for train_index, test_index in KF.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores_macro.append(f1_score(y_test, y_pred, average="macro"))
        f1_scores_micro.append(f1_score(y_test, y_pred, average="micro"))
    accuracies = np.array(accuracies)
    f1_scores_macro = np.array(f1_scores_macro)
    f1_scores_micro = np.array(f1_scores_micro)
        
    names = ['none_X_BOW', 'porter_X_BOW', 'snowball_X_BOW', 'lemmatized_X_BOW', 'none_X_tfidf', 'porter_X_tfidf', 'snowball_X_tfidf', 'lemmatized_X_tfidf', 'none_X_lsa', 'porter_X_lsa', 'snowball_X_lsa', 'lemmatized_X_lsa']
    print("\nDataset: " + names[i])
    print("Accuracy: %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std() * 2))
    print("F1 Macro: %0.4f (+/- %0.4f)" % (f1_scores_macro.mean(), f1_scores_macro.std() * 2))
    print("F1 Micro: %0.4f (+/- %0.4f)" % (f1_scores_micro.mean(), f1_scores_micro.std() * 2))
    print("")


