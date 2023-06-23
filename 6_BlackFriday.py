import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

lr = LogisticRegression(random_state=1)

test_df = pd.read_csv("E:/Data Science Intership/BF_test.csv")
train_df = pd.read_csv("E:/Data Science Intership/BF_train.csv")

print(test_df)

test_df.head()

print(test_df)

print(test_df.isnull().sum())

print(train_df.isnull().sum())

test_df['Product_Category_3'].fillna((test_df['Product_Category_3']).mean(), inplace = True)
test_df['Product_Category_2'].fillna((test_df['Product_Category_2']).mean(), inplace = True)
train_df['Product_Category_3'].fillna((train_df['Product_Category_3']).mean(), inplace = True)
train_df['Product_Category_2'].fillna((train_df['Product_Category_2']).mean(), inplace = True)

print(test_df.isnull().sum())
print(train_df.isnull().sum())

X = train_df.drop('Product_ID', axis = 1)
X = X.drop('Marital_Status', axis =1)
Y = train_df['Purchase']

print(X.isnull().sum())

combined_df = pd.concat([train_df, test_df])
print(combined_df)

le = LabelEncoder()
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
for col in categorical_cols:
    combined_df[col] = le.fit_transform(combined_df[col])

train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

X_train = train_df.drop('Purchase', axis=1)
X_train = X_train.drop('Product_ID', axis=1)
y_train = train_df['Purchase']

lreg = LinearRegression()
lreg.fit(X_train,y_train)

X_test = test_df.drop('Purchase', axis=1)
X_test = X_test.drop('Product_ID', axis=1)
y_pred = lreg.predict(X_test)

print("-------------------------------------------------------------")
print("Predicted Purchase Amounts:")
print(y_pred)