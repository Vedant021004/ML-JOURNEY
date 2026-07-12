import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv")
# print(df.head(10))
print(df.columns)
print(df.dtypes)