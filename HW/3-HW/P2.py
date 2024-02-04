

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
df = pd.read_csv("reduced_huffpost.csv", encoding="latin-1") #encoding ensures special characters are being read in correctly
labels = df["category"]
df["headline"] = df["headline"].str.strip("b\'\"")
df["short_description"] = df["short_description"].str.strip("b\'\"")
documents = df["headline"] + df["short_description"]


print("Categories:", labels.unique())  

# b. How many instances of each class are there?
print("\nInstances of each class:\n", labels.value_counts()) 

# c. What type of dataset are we dealing with based on these numbers?
instances = labels.value_counts()
if any(instances / instances.sum() < 0.05):
    print("\nThe dataset is imbalanced.")
else:
    print("\nThe dataset is balanced.")