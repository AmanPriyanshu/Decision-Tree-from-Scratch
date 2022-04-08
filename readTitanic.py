import pandas as pd
import numpy as np

def read(path="Titanic.csv"):
	df = pd.read_csv(path)
	df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
	df.Age = ['child' if i<18 else 'young' if i<30 else 'middle_aged' if i<60 else 'old' for i in df.Age]
	df.Fare = ['below_100' if i<100 else 'below_150_above_100' if i<150 else 'below_200_above_150' if i<200 else 'above_200' for i in df.Fare]
	features = df.columns
	df = df.dropna()
	df = df.values
	df = pd.DataFrame(df)
	df.columns = features
	features = list(df.columns)[1:]
	df = df.values
	df = df.astype('str')
	target = df.T[0]
	df = df.T[1:].T
	return df, target, features

if __name__ == '__main__':
	read()