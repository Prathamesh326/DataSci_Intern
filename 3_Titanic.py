import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 1)
gbn = GradientBoostingClassifier(n_estimators = 10)
df = pd.read_csv("E:/Data Science Intership/Titanic-Dataset.csv")
#print(df)
df['Age'].fillna((df['Age']).mean(), inplace = True)
X = df.drop('Name', axis = 1)
X = X.drop('PassengerId', axis =1)
X = X.drop('Cabin', axis =1)
X = X.drop('SibSp', axis =1)
X = X.drop('Parch', axis =1)
X = X.drop('Ticket', axis =1)
X = X.drop('Fare', axis =1)
# X = X.drop('Embarked', axis =1)
# print(X.isnull().sum())

Y = df['Survived']


#print(df.isnull().sum())
#print(df['Age'])

# Label encoder
le = LabelEncoder()
le.fit(X['Sex'])
X['Sex']=le.transform(X['Sex'])
#print(X.isnull().sum())

le = LabelEncoder()
le.fit(X['Embarked'])
X['Embarked']=le.transform(X['Embarked'])
# print(df['Age'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=1,test_size = 0.2)

#Random Forest Classifier
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, Y_train)
y_pred=rf.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy RF:", accuracy)

#Logistic Regression
lr.fit(X_train, Y_train)
y_pred=lr.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy LR:", accuracy)

#MultinomialNB
nb.fit(X_train, Y_train)
y_pred=nb.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy NB:", accuracy)

#DecisionTree
dt.fit(X_train, Y_train)
y_pred=dt.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy dt:", accuracy)

#GradientBoosting
gbn.fit(X_train, Y_train)
y_pred=gbn.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy gbn:", accuracy)




