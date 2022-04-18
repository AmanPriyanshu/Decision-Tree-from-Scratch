import numpy as np
from reader import read
from readTitanic import read

class Node:
	def __init__(self, attribute_name, value):
		self.attribute_name = attribute_name
		self.value = value
		self.p_vals = None
		self.children = []

	def add_children(self, attribute_name, value):
		child = Node(attribute_name, value)
		self.children.append(child)

	def predict(self, sample):
		if self.attribute_name!='True' and sample[self.attribute_name]!=self.value:
			return False
		if len(self.children)==0:
			return self.p_vals
		else:
			for child in self.children:
				response = child.predict(sample)
				if not response:
					continue
				else:
					return response

class DecisionTree:
	def __init__(self, max_depth=10, verbose=True):
		self.max_depth = max_depth
		self.root = None
		self.verbose = verbose
		self.fit = self.make_split

	def get_attribute_categories(self, data):
		return [[i for i in np.unique(col)] for col in data.T]

	def compute_attribute_wise_entropy(self, data, target):
		attribute_categories = self.get_attribute_categories(data)
		s_arr = []
		for attribute_idx, attribute in enumerate(attribute_categories):
			s = 0
			for cat in attribute:
				indices = np.argwhere(data.T[attribute_idx]==cat).flatten()
				sub_data = data[indices]
				sub_target = target[indices]
				num_samples = indices.shape[0]
				count_array = np.unique(sub_target, return_counts=True)[1]
				p = count_array/np.sum(count_array)
				s_v = - np.sum(p*np.log2(p))
				p_v = num_samples/data.shape[0]
				s+=p_v*s_v
			s_arr.append(s)
		return s_arr

	def make_split(self, data, target, attribute_names, tab="", node=None):
		s_arr = self.compute_attribute_wise_entropy(data, target)
		split_index = s_arr.index(max(s_arr))
		attribute_categories = self.get_attribute_categories(data)
		for val_cat in attribute_categories[split_index]:
			sub_data_indices = np.argwhere(data.T[split_index]==val_cat).flatten()
			sub_data = data[sub_data_indices]
			sub_target = target[sub_data_indices]
			sub_data = np.delete(sub_data, split_index, 1)
			if node is None:
				self.root = Node("True", "True")
				node = self.root
				self.default = target[0]
			node.add_children(attribute_names[split_index], val_cat)
			if self.verbose:
				print(tab+"`->"+attribute_names[split_index]+"=="+val_cat)
			if sub_data.shape[1]==0:
				node.children[-1].p_vals={i:str(round(100*j/sub_target.shape[0], 1))+"%" for i,j in zip(np.unique(sub_target, return_counts=True)[0], np.unique(sub_target, return_counts=True)[1])}
				if self.verbose:
					print(tab+"   `->"+str({i:str(round(100*j/sub_target.shape[0], 1))+"%" for i,j in zip(np.unique(sub_target, return_counts=True)[0], np.unique(sub_target, return_counts=True)[1])}))
			else:
				sub_attribute_names = attribute_names[:]
				sub_attribute_names.pop(split_index)
				self.make_split(sub_data, sub_target, sub_attribute_names, tab=tab+"   ", node=node.children[-1])

	def score(self, samples, target):
		acc = 0
		for sample, label in zip(samples, target):
			response = dt.root.predict(sample)
			classes, preds = [], []
			if response is not None:
				for key, item in response.items():
					classes.append(key)
					preds.append(float(item.replace('%', '')))
				pred = classes[preds.index(max(preds))]
			else:
				pred = self.default
			if pred==label:
				acc+=1
		return acc/len(samples)

if __name__ == '__main__':
	split = 0.9
	dt = DecisionTree(verbose=True)
	data, target, attribute_names = read()
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	target = target[indices]
	train_data = data[:int(split*data.shape[0])]
	train_target = target[:int(split*data.shape[0])]
	test_data = data[int(split*data.shape[0]):]
	test_target = target[int(split*data.shape[0]):]
	dt.fit(train_data, train_target, attribute_names)
	samples = [{name:val for name,val in zip(attribute_names, row)} for row in train_data]
	accuracy = dt.score(samples, train_target)
	print(f"\n\nTrain Performance: {str(round(accuracy, 2))}")
	samples = [{name:val for name,val in zip(attribute_names, row)} for row in test_data]
	accuracy = dt.score(samples, test_target)
	print(f"\n\nTest Performance: {str(round(accuracy, 2))}")