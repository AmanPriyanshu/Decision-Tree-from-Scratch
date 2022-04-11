import pandas as pd
import numpy as np

def read(path="heart_2020_cleaned.csv", N=50000, k_health=6, k_time=4):
	df = pd.read_csv(path)
	df.BMI = [str(10*int(i//10))+'-'+str(int(10*(1+int(i//10)))) for i in df.BMI]
	df.PhysicalHealth = [str(k_health*int(i//k_health))+'-'+str(k_health+k_health*int(i//k_health)) for i in df.PhysicalHealth]
	df.MentalHealth = [str(k_health*int(i//k_health))+'-'+str(k_health+k_health*int(i//k_health)) for i in df.MentalHealth]
	df.SleepTime = [str(k_time*int(i//k_time))+'-'+str(k_time+k_time*int(i//k_time)) for i in df.SleepTime]
	features = df.columns
	df = df.values
	df = df[:N]
	target = df.T[0]
	df = df.T[1:].T
	features = list(features)
	features = features[1:]
	return df, target, features

if __name__ == '__main__':
	read()