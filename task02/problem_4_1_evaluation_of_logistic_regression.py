# Package imports
import seaborn as sns

sns.set()
sns.set_style('darkgrid')
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                           weights=None, flip_y=0.01, class_sep=0.85, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=0)

# Split the dataset into training and validation segments
X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=1234)

# Train the logistic regression linear classifier
clf = LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                           fit_intercept=True, intercept_scaling=1.0, max_iter=200,
                           multi_class='multinomial', n_jobs=1, penalty='l2', random_state=0,
                           refit=True, scoring=None, solver='sag', tol=0.0001, verbose=0)
clf.fit(X, y)

# Evaluate logistic classifier on validation set
clf_preds = clf.predict(X_val)

print('\nClassification Report for Logistic Regression:\n')
print(classification_report(y_val, clf_preds))
print('Validation Accuracy for Logistic Regression: {:.6f}'.format(np.mean(y_val == clf_preds)))
