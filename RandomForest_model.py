from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')


RF_model = RandomForestClassifier(n_estimators=1000,random_state=0)


x_train = train_set.iloc[:,0:12].values
y_train = train_set.iloc[:,12].values

x_test = test_set.iloc[:,0:12].values
y_test = test_set.iloc[:,12].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

RF_model.fit(x_train,y_train)
y_prediction = RF_model.predict(x_test)
print("GHOSTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\n\n")
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))