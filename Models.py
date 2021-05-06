def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, average_precision_score, f1_score, precision_score
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

train_set = pd.read_csv('train_10_only.csv')
test_set = pd.read_csv('test_10_only.csv')


print("\n\n RandomForestClassifier with StandardScaler n_estimators=1200:")
RF_model = RandomForestClassifier(n_estimators=1200, random_state=0)
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
RF_model.fit(x_train, y_train)
y_prediction = RF_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


print("\n\n RandomForestClassifier without StandardScaler n_estimators=1200:")
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
RF_model.fit(x_train, y_train)
y_prediction = RF_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


print("\n\nKNNeighborsClassifier with StandardScaler n_neighbors=750:")
KNN_model = KNeighborsClassifier(n_neighbors=750)
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
KNN_model.fit(x_train, y_train)
y_prediction = KNN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


print("\n\nKNNeighborsClassifier without StandardScaler n_neighbors=750:")
KNN_model = KNeighborsClassifier(n_neighbors=750)
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
KNN_model.fit(x_train, y_train)
y_prediction = KNN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


print("\n\nSVM model with StandardScaler:")
svm_model = svm.SVC(kernel='rbf')
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
svm_model.fit(x_train, y_train)
y_prediction = svm_model.predict(x_test)
prediction = [round(val) for val in y_prediction]
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


print("\n\nSVM model without StandardScaler:")
svm_model = svm.SVC(kernel='rbf')
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
svm_model.fit(x_train, y_train)
y_prediction = svm_model.predict(x_test)
prediction = [round(val) for val in y_prediction]
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nGaussianNB model with StandardScaler:")
GBN_model = GaussianNB()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
GBN_model.fit(x_train, y_train)
y_prediction = GBN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nGaussianNB model without StandardScaler:")
GBN_model = GaussianNB()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
GBN_model.fit(x_train, y_train)
y_prediction = GBN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nLinearDiscriminantAnalysis model with StandardScaler:")
LDA_model = LinearDiscriminantAnalysis()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
LDA_model.fit(x_train, y_train)
y_prediction = LDA_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nLinearDiscriminantAnalysis model without StandardScaler:")
LDA_model = LinearDiscriminantAnalysis()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
LDA_model.fit(x_train, y_train)
y_prediction = LDA_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nLogistic Reg model with StandardScaler:")
LR_model = LogisticRegression()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
LR_model.fit(x_train, y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print(
    "\n\nLogistic Reg model with StandardScaler random_state=1, max_iter=150, penalty=l2, C=0.275, solver='saga', multi_class='ovr':")
LR_model = LogisticRegression(random_state=1, max_iter=150, penalty="l2", C=0.275, solver='saga', multi_class='ovr')
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
LR_model.fit(x_train, y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nLogistic Reg model without StandardScaler:")
LR_model = LogisticRegression()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
LR_model.fit(x_train, y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nXGBoost model with StandardScaler:")
XGB_model = XGBClassifier()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
XGB_model.fit(x_train, y_train)
y_prediction = XGB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nXGBoost model without StandardScaler:")
XGB_model = XGBClassifier()
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
XGB_model.fit(x_train, y_train)
y_prediction = XGB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nAdaBoost model with StandardScaler:")
AdaB_model = AdaBoostClassifier(n_estimators=200, random_state=2)
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
AdaB_model.fit(x_train, y_train)
y_prediction = AdaB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

print("\n\nAdaBoost model without StandardScaler:")
AdaB_model = AdaBoostClassifier(n_estimators=200, random_state=2)
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
AdaB_model.fit(x_train, y_train)
y_prediction = AdaB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))
