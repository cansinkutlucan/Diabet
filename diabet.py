import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.eda import *
from helpers.data_prep import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("diabetes.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0,0.01, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numeric variables analysis
for col in num_cols:
    num_summary(df, col, plot=True)

# Numeric variables analysis by target variable
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

############################################
# Data Preprocessing
############################################
NaN_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome","Age"])]

for col in NaN_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()

na_columns = missing_values_table(df, na_name=True)

missing_vs_target(df, "Outcome", na_columns)


# Filling in missing Values by median in categorical variable breakdown 
def median_target(col):
    temp = df[df[col].notnull()]
    temp = temp[[col, 'Outcome']].groupby(['Outcome'])[[col]].median().reset_index()
    return temp

for col in NaN_columns:
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]

df.isnull().sum()

# Replace outliers to a certain value
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

check_df(df)

############################################
# Feature Engineering
############################################
AGE_CAT = pd.Series(['Young','Middle_Aged','Old'],dtype = "category")

df["AGE_CAT"] = AGE_CAT
df.loc[df["Age"] <= 44, "AGE_CAT"] = AGE_CAT[0]
df.loc[(df["Age"] > 44) & (df["BMI"] <= 65), "AGE_CAT"] = AGE_CAT[1]
df.loc[df["Age"] > 65, "AGE_CAT"] = AGE_CAT[2]

NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"],
                   dtype = "category")

df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]

NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")

df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]

NewBloodPressure = pd.Series(["Low", "Normal", "High"], dtype = "category")
df["NewBloodPressure"] = NewBloodPressure
df.loc[df["BloodPressure"] <= 60, "NewBloodPressure"] = NewBloodPressure[0]
df.loc[(df["BloodPressure"] > 60) & (df["BloodPressure"] <= 90), "NewBloodPressure"] = NewBloodPressure[1]
df.loc[(df["BloodPressure"] > 90), "NewBloodPressure"] = NewBloodPressure[2]

df["age_weighted_dpf"] = df["DiabetesPedigreeFunction"] * df["Age"] / 15

def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df.columns = [col.upper() for col in df.columns]

# LABEL ENCODING
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# ONE-HOT ENCODING
df = pd.get_dummies(df, drop_first=True)
df.head()

# MODELLING
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

lgr = LogisticRegression(random_state=12345)
lgr_model = lgr.fit(X_train, y_train)

# TEST ERROR
y_pred = lgr_model.predict(X_test)
y_prob = lgr_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy_score(y_test, y_pred)
# 0,85

# Precision
precision_score(y_test, y_pred)
# 0,82

# Recall
recall_score(y_test, y_pred)
# 0,74

# F1
f1_score(y_test, y_pred)
# 0,78

print(classification_report(y_test, y_pred))

# AUC
roc_auc_score(y_test, y_prob)
# 0,91

random_user = X.sample(1)
lgr_model.predict(random_user)

# Yes, we may use other ML techniques, hyperparameter optimization etc. to make a better prediction
