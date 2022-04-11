from trees import DecisionTree
from readHeart import read
import numpy as np
from tqdm import tqdm

class RandomForest:
	def __init__(self, num_trees):
		self.num_trees = num_trees
		self.seed = 0
		np.random.seed(self.seed)
		self.forest = [DecisionTree(verbose=False) for _ in range(self.num_trees)]

	def set_index_distribution(self, target):
		all_indices = np.arange(target.shape[0])
		np.random.seed(self.seed)
		sizes = np.random.randint(low=int(0.25*all_indices.shape[0]), high=int(0.75*all_indices.shape[0]), size=self.num_trees)
		self.index_distribution = [np.random.choice(all_indices, size=sizes[i]) for i in range(self.num_trees)]

	def fit(self, data, target, attribute_names):
		self.set_index_distribution(target)
		[tree.fit(data, target, attribute_names) for tree in tqdm(self.forest, desc="Training")]	

	def score(self, samples, target):
		scores = [tree.score(samples, target) for tree in tqdm(self.forest, desc="Predicting")]
		return sum(scores)/len(scores)

if __name__ == '__main__':
	split = 0.8
	data, target, attribute_names = read()
	indices = np.arange(data.shape[0])
	np.random.seed(0)
	np.random.shuffle(indices)
	data = data[indices]
	target = target[indices]
	train_data = data[:int(split*data.shape[0])]
	train_target = target[:int(split*data.shape[0])]
	test_data = data[int(split*data.shape[0]):]
	test_target = target[int(split*data.shape[0]):]

	rf = RandomForest(10)
	rf.fit(train_data, train_target, attribute_names)
	samples = [{name:val for name,val in zip(attribute_names, row)} for row in train_data]
	accuracy = rf.score(samples, train_target)
	print("\n\nTrain Performance:", str(round(accuracy, 2))+"%")
	samples = [{name:val for name,val in zip(attribute_names, row)} for row in test_data]
	accuracy = rf.score(samples, test_target)
	print("\n\nTest Performance:", str(round(accuracy, 2))+"%")