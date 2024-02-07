'''
This week we are performing sentiment analysis on Rotten Tomatoes movie reviews (published since 2023) using a sentiment lexicon (SentiWordNet and VADER), and different classification models. I am not providing code to read in the CSV file. You will have to read the file and select the movie reviews as the feature data and the associated sentiment as the labels. There will be two different approaches in this homework: 
•	Unsupervised learning, where we pretend we do not have any labels and apply a sentiment lexicon to auto-label the data (but we will compare the lexicon output to the actual labels)
•	Supervised learning using different machine learning models
As in the last homework, we will be applying different stemming/lemmatization and vectorization techniques, and calculating accuracy and F1-scores.
NOTE: the sentiment labels in this data are “POSITIVE” and “NEGATIVE”, meaning there is no “NEUTRAL” category, as a result, we will not be classifying anything as neutral either.


Part 1 – Unsupervised learning
Write code that accomplishes the following for ALL reviews in the dataset (i.e., NO train-test split):
1.	Apply VADER polarity analysis on each review
2.	Apply SentiWordNet analysis on each review
    a.	Apply POS-tagging and label each word in the data using the sentiwordnet lexicon
    b.	Assign an overall sentiment based on the word labels
3.	Decide whether to classify ‘0’ scores as positive or negative
4.	Calculate the accuracy and f1-scores by comparing the two lexicon labels to the original labels
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


'''


# Importing the necessary libraries
import pandas as pd
import nltk
from nltk.corpus import reuters
from nltk import WordNetLemmatizer
from nltk.sentiment import *
from nltk.corpus import wordnet, sentiwordnet
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("rotten_tomatoes_reviews.csv", encoding="latin-1") #encoding ensures special characters are being read in correctly
reviews = df["reviewText"]
original_labels = df["scoreSentiment"]#based off the file has originalScore as the sentiment label with POSTIVE and NEGATIVE

vader_labels = []
sentiwordnet_labels = []

# Sentiment Analysis object
sia = SentimentIntensityAnalyzer()

# Part 1 – Unsupervised learning
def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''



def Part_one():
        # Part 1 – Unsupervised learning
    for review in reviews:
        # Apply VADER polarity analysis on each review
        vader_sentiment = sia.polarity_scores(review)
        vader_labels.append('POSITIVE' if vader_sentiment['compound'] >= 0 else 'NEGATIVE')

        # Apply SentiWordNet analysis on each review
        pos_scores = []
        neg_scores = []

        tokens = word_tokenize(review)
        tagged = nltk.pos_tag(tokens)

        for word, tag in tagged:
            wn_tag = wordnet_pos_code(tag)
            if wn_tag != '':
                synsets = list(sentiwordnet.senti_synsets(word, wn_tag))
                if len(synsets) > 0:
                    pos_score = synsets[0].pos_score()
                    neg_score = synsets[0].neg_score()
                    pos_scores.append(pos_score)
                    neg_scores.append(neg_score)

        # Assign an overall sentiment based on the word labels
        avg_pos_score = sum(pos_scores) / len(pos_scores) if pos_scores else 0
        avg_neg_score = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    
        sentiwordnet_labels.append('POSITIVE' if avg_pos_score > avg_neg_score else 'NEGATIVE')




    # Calculate accuracy and F1-score for Vader
    vader_accuracy = accuracy_score(original_labels, vader_labels)
    vader_f1_score = f1_score(original_labels, vader_labels, average='binary', pos_label='POSITIVE')

    print(f'VADER - Accuracy: {vader_accuracy}, F1-Score: {vader_f1_score}')

    # Calculate accuracy and F1-score for SentiWordNet
    swn_accuracy = accuracy_score(original_labels, sentiwordnet_labels)
    swn_f1_score = f1_score(original_labels, sentiwordnet_labels, average='binary', pos_label='POSITIVE')

    print(f'SentiWordNet - Accuracy: {swn_accuracy}, F1-Score: {swn_f1_score}')
    
    # end function part 1
    
    
Part_one() # calling the function to run the code
    
    
    



    
