import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

dt = pd.read_csv('iris.csv')

X = dt.iloc[:,:-1].values
y = dt.variety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

filename = 'iris_SVC_model.z'
joblib.dump(model, filename)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))