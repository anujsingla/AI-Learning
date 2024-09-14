# https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python/notebook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
import category_encoders as ce


data = '<file path>'
df = pd.read_csv(data, header=None, sep=',')
# print(df.shape)

# print(df.head())
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

# print(df.columns)
# print(df.head())
# print(df.info())

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

# print('There are {} categorical variables\n'.format(len(categorical)))

# print('The categorical variables are :\n\n', categorical)
# print(df[categorical].head())
# check missing values in categorical variables
# print(df[categorical].isnull().sum())

# view frequency counts of values in categorical variables

# for var in categorical: 
#     print(df[var].value_counts())

# check labels in workclass variable
# print(df.workclass.unique())

# check frequency distribution of values in workclass variable
#print(df.workclass.value_counts())

# replace '?' values in workclass variable with `NaN`
# Strip leading/trailing whitespaces (if any)
df['workclass'] = df['workclass'].str.strip()
# # Ensure the column is treated as a string
df['workclass'] = df['workclass'].astype(str)
# # Replace '?' with np.nan
df['workclass'] = df['workclass'].replace('?', np.nan)

# print(df.workclass.value_counts())

# Strip leading/trailing whitespaces (if any)
df['occupation'] = df['occupation'].str.strip()
# # # Ensure the column is treated as a string
df['occupation'] = df['occupation'].astype(str)
# # # Replace '?' with np.nan
df['occupation'] = df['occupation'].replace('?', np.nan)
# print(df.occupation.value_counts())


# Strip leading/trailing whitespaces (if any)
df['native_country'] = df['native_country'].str.strip()
# # # # Ensure the column is treated as a string
df['native_country'] = df['native_country'].astype(str)
# # # # Replace '?' with np.nan
df['native_country'] = df['native_country'].replace('?', np.nan)
# # print(df.occupation.value_counts())
# print(df.native_country.value_counts())

# print(df[categorical].isnull().sum())

X = df.drop(['income'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Print the results to verify the split
# print("X_train:\n", X_train)
# print("X_test:\n", X_test)
# print("y_train:\n", y_train)
# print("y_test:\n", y_test)

# print(X_train.shape, X_test.shape)

# 10. Feature Engineering 

# print(X_train.dtypes)
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

print(categorical)

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

# print(numerical)

# print(X_train[categorical].isnull().mean())
# print(X_test[categorical].isnull().sum())
# print(X_train.isnull().sum())
print(X_train[categorical].head())

# encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
# print(X_train.head())

# 11. Feature Scaling

cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())

# 12. Model training
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)

# 13. Predict the results
y_pred = gnb.predict(X_test)
print(y_pred)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print("end of Naive Bayes")