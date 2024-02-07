'''
Part 2 – Supervised learning
Write code that accomplishes the following (feel free to create a new python file):
1.	Apply a stemming technique of your choice, as well as the wordnet lemmatizer (= two different techniques)
2.	Apply different vectorization techniques
    a.	Filter out duplicate words and send the resulting document through sklearn’s CountVectorizer
    b.	Tf-idf vectorizer
3.	Train three different machine learning models using 10-fold CV, must include Naïve Bayes, and choose two more from the following options:
    a.	Logistic Regression
    b.	Decision Tree
    c.	Support Vector Machine
    d.	Random Forest
Be careful which ones you choose, you will be expected to think critically about the performance of these models. 
4.	Evaluate the models using accuracy and F1-score (micro and macro).





I have all of the available models in the code below. I have also included the execution time for each model as I was curious how long each one would take to run.
They are running in parrallel so the time is not the total time for all models to run, it is the time for each model to run. It took a little bit of time to run, but it does run. 
There should not be any errors with its functionality in finishing the code.


'''

# Importing the necessary libraries
import pandas as pd
import numpy as np
import time
from nltk import WordNetLemmatizer
from nltk.sentiment import *
from nltk.corpus import wordnet, sentiwordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv("rotten_tomatoes_reviews.csv", encoding="latin-1") #encoding ensures special characters are being read in correctly
reviews = df["reviewText"]
original_labels = df["scoreSentiment"]#based off the file has originalScore as the sentiment label with POSTIVE and NEGATIVE

stop_words = set(stopwords.words('english'))

vader_labels = []
sentiwordnet_labels = []

LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform(original_labels) 
# lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_docs = []


for review in reviews:
    token_words = word_tokenize(review)
    lemmatized_sentence = ' '.join(lemmatizer.lemmatize(word) for word in token_words if word.lower() not in stop_words)
    lemmatized_docs.append(lemmatized_sentence)
    
    
     
#Vectorizer TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(lemmatized_docs)

# Filter out duplicate words
filtered_docs = [' '.join(list(dict.fromkeys(word_tokenize(doc)))) for doc in lemmatized_docs]

# Apply CountVectorizer
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(filtered_docs)


# Train three different machine learning models using 10-fold CV, must include Naïve Bayes, and choose two more 

nb_model = MultinomialNB()#1
lr_model = LogisticRegression()#2
svm_model = SVC()#3
rf_model = RandomForestClassifier()#4
dt_model = DecisionTreeClassifier()#I was curious how this would perform so i added it as well.

models = [nb_model, lr_model, svm_model, rf_model, dt_model]
model_names = ["Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree"]

for model, name in zip(models, model_names):
    accuracies = []
    f1_scores_macro = []
    f1_scores_micro = []
    for X in [X_tfidf, X_count]:
        start_time = time.time()
        metrics = cross_validate(model, X, y, cv=10, scoring=['accuracy', 'f1_macro', 'f1_micro'], n_jobs=-1, return_train_score=False)
        end_time = time.time()
        accuracies.append(metrics['test_accuracy'].mean())
        f1_scores_macro.append(metrics['test_f1_macro'].mean())
        f1_scores_micro.append(metrics['test_f1_micro'].mean())
    accuracies = np.array(accuracies)
    f1_scores_macro = np.array(f1_scores_macro)
    f1_scores_micro = np.array(f1_scores_micro)
    
    print("\nModel: " + name)
    print("Execution Time: %s seconds" % (end_time - start_time))
    print("Accuracy: %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std() * 2))
    print("F1 Macro: %0.4f (+/- %0.4f)" % (f1_scores_macro.mean(), f1_scores_macro.std() * 2))
    print("F1 Micro: %0.4f (+/- %0.4f)" % (f1_scores_micro.mean(), f1_scores_micro.std() * 2))
    print("")



