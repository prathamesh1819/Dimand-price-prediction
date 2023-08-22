import numpy as np
import pandas as pd

df = pd.read_csv('./Salary_Data.csv')

x = df.drop('Salary', axis=1)
y = df['Salary']

# regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)

# save the model
import pickle
with open('mode.pkl', 'wb') as file:
    pickle.dump(model, file)
