import numpy as np

num_recipes = 45389

class CTRMF():
	def __init__(self,
				 ratings_data,
				 recipe_feature,
				 n_hidden=100,
				 reg_term=0.01,
				 verbose=False):
		"""
		"""
		self.ratings_data = ratings_data # users x recipes
		self.recipe_feature = recipe_feature # recipes x features
		self.n_hidden = n_hidden
		self.n_users, self.n_recipes = ratings_data.shape
		self.n_features = recipe_feature.shape[1]
		self.reg_term = reg_term
		self.ratings_users, ratings_recipes = self.ratings_data.nonzero()
		self.n_samples = len(self.ratings_users)
		self.verbose = verbose

	def train(self, n_iter=10, learning_rate=0.1):
		self.user_hidden    = np.random.normal(scale=1./self.n_hidden,
									           size=(self.n_users, self.n_hidden))
		self.hidden_feature = np.random.normal(scale=1./self.n_hidden,
											   size=(self.n_hidden, self.n_features))
		self.learning_rate = learning_rate
		self.user_bias = np.zeros(self.n_users)
		self.recipe_bias = np.zeros(self.n_recipes)
		self.global_bias = np.mean(self.ratings_data[np.where(self.ratings_data != 0)])
		self.partial_train(n_iter)
	
	def partial_train(self, n_iter):
		for i in range(n_iter):
			if i % 10 == 0 and self.verbose:
				print('\tcurrent iteration: {}'.format(i))
			self.training_indicies = np.arange(self.n_samples)
			np.random.shuffle(self.training_indices)
			self.sgd()

	def sgd(self):
		"""
		Perform stochastic gradient descent
		"""
		for index in self.training_indices:
			u = self.ratings_users[index]
			r = self.ratings_recipes[index]
			prediction = self.predict(u, r)
			e = error(u,r)
			self.hidden_feature += self.learning_rate * \
				sum([self.error(u,x) * np.outer(self.user_hidden, self.recipe_feature[x]) for x in range(n_recipes)])
			self.recipe_bias[r] += self.learning_rate * (e - self.reg_term * self.recipe_bias[r])
			self.user_bias[u] += self.learning_rate * (e - self.reg_term * self.user_bias[u])
			self.user_hidden[u] += self.learning_rate * \
				sum([self.error(u,x) * self.hidden_feature.dot(self.recipe_feature[x]) - self.reg_term * self.user_hidden[u] for x in range(n_recipes)])
	
	def predict(self, u, r):
		prediction = self.global_bias + self.user_bias[u] + self.recipe_bias[r]
		prediction += self.user_hidden[u,:].dot(self.hidden_feature).dot(self.recipe_feature[r,:])
		return prediction
	
	def error(self, u, r):
		return self.ratings_data[u,r] - predict(u,r)


