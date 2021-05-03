from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

cols_names = ['home_team_mean_rating', 'away_team_mean_rating',
                   'home_top_players',
                   'away_top_players', 'AVG_bet_D', 'AVG_bet_H', 'AVG_bet_A', 'h_5_games_pts', 'a_5_games_pts',
                   'buildUpPlaySpeedHome', 'defencePressureHome', 'defenceAggressionHome', 'chanceCreationPassingHome',
                   'chanceCreationShootingHome', 'buildUpPlaySpeedAway', 'defencePressureAway', 'defenceAggressionAway',
                   'chanceCreationPassingAway', 'chanceCreationShootingAway']

train_set = pd.read_csv('./with_team_att_dor/train_5_games.csv')
test_set = pd.read_csv('./with_team_att_dor/test_5_games.csv')


RF_model = RandomForestClassifier(n_estimators=1000,random_state=0)


x_train = train_set.iloc[:,3:22].values
y_train = train_set.iloc[:,22].values

x_test = test_set.iloc[:,3:22].values
y_test = test_set.iloc[:,22].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

RF_model.fit(x_train,y_train)
y_prediction = RF_model.predict(x_test)
print("GHOSTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\n\n")
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))


feature_importance = RF_model.feature_importances_
zip_name = zip(cols_names,feature_importance)
zip_name = sorted(list(zip_name),key=lambda x: x[1], reverse=True)
plt.bar(range(len(zip_name)), [val[1] for val in zip_name],align='center')
plt.xticks(range(len(zip_name)), [val[0] for val in zip_name])
plt.xticks(rotation=90)
plt.title("feature importance of Random Forest")
plt.tight_layout()
plt.show()