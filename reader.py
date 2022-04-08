import pandas as pd

def read(path="Covid.csv"):
	df = pd.read_csv(path)
	df = df.drop(['ID'], axis=1)
	attributes = list(df.columns)[:-1]
	df = df.values
	return df.T[:-1].T, df.T[-1], attributes