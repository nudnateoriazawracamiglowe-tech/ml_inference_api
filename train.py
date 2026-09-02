from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

#loading the dataset
#X train set
#Y test set
data=load_breast_cancer()
X = data.data
y = data.target
print(X.shape)
print(y.shape)
print(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
#classification wheter or not the tumor is bening or malignant
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
accuracy=accuracy_score(y_test, predictions)
print("the accuracy score is:",{accuracy})
filename = 'svm_classification.joblib'
joblib.dump(svm, filename)

print(f"Model saved to {filename}")