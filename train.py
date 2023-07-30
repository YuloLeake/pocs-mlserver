import pickle

import joblib
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)
print(y)
print(iris.feature_names)
print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 4)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(knn.predict_proba(X_test))

# Save model using pickle
with open('iris.pickle', 'wb') as f:
    pickle.dump(knn, f)

# Save model using joblib
joblib.dump(knn, "iris.joblib")
