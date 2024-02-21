import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
# label encoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# load data set
df = pd.read_csv("rotten_tomatoes_reviews.csv", encoding="latin-1")
reviews = df["reviewText"]
sentiment = df["scoreSentiment"]

bow = CountVectorizer()
X = bow.fit_transform(reviews)
y = LabelEncoder().fit_transform(sentiment)

# model
rf = RandomForestClassifier()


# parameters
param_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}

# grid search
cv = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
cv.fit(X, y)

print("Best parameters:", cv.best_params_)
print("Best score:", cv.best_score_)

# Evaluate the best model on the testing set
best_model = cv.best_estimator_
accuracy = best_model.score(X, y)
print("Testing accuracy:", accuracy)

# Plot the cross-validated scores for different parameter combinations
results = pd.DataFrame(cv.cv_results_)
plt.figure(figsize=(8, 6))
for params, mean_score in zip(results['params'], results['mean_test_score']):
    plt.plot(params['rf__n_estimators'], mean_score, 'bo')
    plt.xlabel('Number of estimators')
    plt.ylabel('Mean cross-validated score')
plt.show()
