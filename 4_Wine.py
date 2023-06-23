import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df = pd.read_csv("E:/Data Science Intership/winequalityN.csv")
# print(df.head(10))

df['fixed acidity'].fillna(df['fixed acidity'].mean(), inplace = True)
df['volatile acidity'].fillna(df['volatile acidity'].mean(), inplace = True)
df['pH'].fillna(df['pH'].mean(), inplace = True)
df.dropna(axis=0, inplace=True)

x = df.drop('quality', axis = 1)
x = x.drop('fixed acidity', axis = 1)
x = x.drop('volatile acidity', axis =1)
x = x.drop('citric acid', axis =1)
#x = x.drop('residual sugar', axis =1)
x = x.drop('chlorides', axis =1)
#x = x.drop('free sulfur dioxide', axis =1)
#x = x.drop('total sulfur dioxide', axis =1)
x = x.drop('density', axis =1)
x = x.drop('sulphates', axis =1)

# x = df[['type','free sulfur dioxide','total sulfur dioxide','alcohol']]              #residual sugar , free sulfur dioxide ,total sulfur dioxide, alcohol, type
y = df['quality']
print(x.keys())
# print(df.isnull().sum())

#Line Encoder
le = LabelEncoder()
le.fit(x['type'])
#x['type']= le.transform(x['type'])
x.loc[:, 'type'] = le.transform(x['type'])


#k best
bestFeatures = SelectKBest(score_func=chi2, k = 'all')
fit = bestFeatures.fit(x,y)                                  #training our model
dfscores = pd.DataFrame(fit.scores_)                         #storing scores
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', ' Score']
print(featureScores)
#print(x.keys())
print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
#RandomForestClassifier
rf = RandomForestClassifier(random_state = 1)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy RF:", accuracy)

'''
#MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy NB:", accuracy)

#LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy lr:", accuracy)
'''