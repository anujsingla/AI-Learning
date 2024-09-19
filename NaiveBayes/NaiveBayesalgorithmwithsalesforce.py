# https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python/notebook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from salesforce_api import get_salesforce_access_token, get_salesforce_report_data, get_salesforce_adults_data

def convert_numeric_columns_to_int(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(int)
    return df

try:
    # Get the access token
    access_token = get_salesforce_access_token()
    print("Access Token:", access_token)

    # Fetch Salesforce adult data
    adult_data = get_salesforce_adults_data(access_token)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(adult_data)
    df = df.drop(columns=['attributes', 'Id'], errors='ignore')
    # print("adult Data:", df)
    # print(print(df.head()))
    # Mapping of Salesforce field names to your desired column names
    field_mapping = {
        'X39__c': 'age',
        'stategov__c': 'workclass',
        'X77516__c': 'fnlwgt',
        'bachelors__c': 'education',
        'X13__c': 'education_num',
        'never_married__c': 'marital_status',
        'Adm_clerical__c': 'occupation',
        'Not_in_family__c': 'relationship',
        'White__c': 'race',
        'Male__c': 'sex',
        'X2174__c': 'capital_gain',
        'X0__c': 'capital_loss',
        'X40__c': 'hours_per_week',
        'United_States__c': 'native_country',
        'X50K__c': 'income'
    }
    df = df.rename(columns=field_mapping)
    df = convert_numeric_columns_to_int(df)
    #print(print(df.head()))
    print(df.shape)
    categorical = [var for var in df.columns if df[var].dtype=='O']
    
    df['workclass'] = df['workclass'].str.strip()
    df['workclass'] = df['workclass'].astype(str)
    df['workclass'] = df['workclass'].replace('?', np.nan)

    # print(df.workclass.value_counts())

    df['occupation'] = df['occupation'].str.strip()
    df['occupation'] = df['occupation'].astype(str)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    # print(df.occupation.value_counts())


    df['native_country'] = df['native_country'].str.strip()
    df['native_country'] = df['native_country'].astype(str)
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

except Exception as e:
    print(f"Error: {e}")

