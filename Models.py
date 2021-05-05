def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import recall_score,average_precision_score,f1_score,precision_score
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


scoring = ["accuracy","recall"]


cols_names = ['home_team_api_id', 'away_team_api_id', 'home_team_mean_rating', 'away_team_mean_rating',
              'home_top_players','away_top_players', 'AVG_bet_H', 'AVG_bet_A_D', 'h_5_games_pts', 'a_5_games_pts',
              'h_only_10_games_pts', 'a_only_10_games_pts']

train_set = pd.read_csv('train_10_only.csv')
test_set = pd.read_csv('test_10_only.csv')



print("\n\n RandomForestClassifier with StandardScaler n_estimators=1200:")
RF_model = RandomForestClassifier(n_estimators=1200 ,random_state=0)
x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

RF_model.fit(x_train,y_train)
y_prediction = RF_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))


print("\n\n RandomForestClassifier without StandardScaler n_estimators=1200:")
x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

RF_model.fit(x_train,y_train)
y_prediction = RF_model.predict(x_test)
# print("GHOSTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\n\n")
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

feature_importance = RF_model.feature_importances_
zip_name = zip(cols_names[3:],feature_importance)
zip_name = sorted(list(zip_name),key=lambda x: x[1], reverse=True)
plt.bar(range(len(zip_name)), [val[1] for val in zip_name],align='center')
plt.xticks(range(len(zip_name)), [val[0] for val in zip_name])
plt.xticks(rotation=90)
plt.title("feature importance of Random Forest")
plt.tight_layout()
plt.show()





print("\n\nKNNeighborsClassifier with StandardScaler n_neighbors=750:")

KNN_model = KNeighborsClassifier(n_neighbors=750)
x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

KNN_model.fit(x_train,y_train)
y_prediction = KNN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))


print("\n\nKNNeighborsClassifier without StandardScaler n_neighbors=750:")

KNN_model = KNeighborsClassifier(n_neighbors=750)
x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

KNN_model.fit(x_train,y_train)
y_prediction = KNN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",average_precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nSVM model with StandardScaler:")

# train_set = pd.read_csv('train.csv')
# test_set = pd.read_csv('test.csv')

svm_model = svm.SVC(kernel='rbf')


x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

scores = cross_validate(svm_model,x_train,y_train,scoring=scoring,cv=5,return_train_score=False)


x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

svm_model.fit(x_train,y_train)
y_prediction = svm_model.predict(x_test)

prediction = [round(val) for val in y_prediction]

print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
# print("Precision, Recall, F-score: \n", precision_recall_fscore_support(y_test,y_prediction))
# print("GGGGGGGGGGGGGGGGGGGG")
# print(scores)
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))



print("\n\nSVM model without StandardScaler:")
# train_set = pd.read_csv('train.csv')
# test_set = pd.read_csv('test.csv')

svm_model = svm.SVC(kernel='rbf')

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

svm_model.fit(x_train,y_train)
y_prediction = svm_model.predict(x_test)
prediction = [round(val) for val in y_prediction]

print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))


print("\n\nGaussianNB model with StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
GBN_model = GaussianNB()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

GBN_model.fit(x_train,y_train)
y_prediction = GBN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nGaussianNB model without StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
GBN_model = GaussianNB()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

GBN_model.fit(x_train,y_train)
y_prediction = GBN_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nLinearDiscriminantAnalysis model with StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
LDA_model = LinearDiscriminantAnalysis()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

LDA_model.fit(x_train,y_train)
y_prediction = LDA_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nLinearDiscriminantAnalysis model without StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
LDA_model = LinearDiscriminantAnalysis()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

LDA_model.fit(x_train,y_train)
y_prediction = LDA_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))


print("\n\nLogistic Reg model with StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
LR_model = LogisticRegression()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

LR_model.fit(x_train,y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nLogistic Reg model with StandardScaler random_state=1, max_iter=150, penalty=l2, C=0.275, solver='saga', multi_class='ovr':")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
LR_model = LogisticRegression(random_state=1, max_iter=150, penalty="l2", C=0.275, solver='saga', multi_class='ovr')


x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

LR_model.fit(x_train,y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nLogistic Reg model without StandardScaler:")
# seed = 42
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
LR_model = LogisticRegression()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

LR_model.fit(x_train,y_train)
y_prediction = LR_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nXGBoost model with StandardScaler:")

XGB_model = XGBClassifier()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

XGB_model.fit(x_train,y_train)
y_prediction = XGB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nXGBoost model without StandardScaler:")

XGB_model = XGBClassifier()

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

XGB_model.fit(x_train,y_train)
y_prediction = XGB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))


print("\n\nAdaBoost model with StandardScaler:")
AdaB_model = AdaBoostClassifier(n_estimators= 200, random_state=2)

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

AdaB_model.fit(x_train,y_train)
y_prediction = AdaB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))

print("\n\nAdaBoost model without StandardScaler:")
AdaB_model = AdaBoostClassifier(n_estimators= 200, random_state=2)

x_train = train_set.iloc[:,3:13].values
y_train = train_set.iloc[:,13].values

x_test = test_set.iloc[:,3:13].values
y_test = test_set.iloc[:,13].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

AdaB_model.fit(x_train,y_train)
y_prediction = AdaB_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
print("Recall score: ",recall_score(y_test, y_prediction, average='macro'))
print("Precision score: ",precision_score(y_test, y_prediction, average='macro'))
print("F1 score: ",f1_score(y_test, y_prediction, average='macro'))