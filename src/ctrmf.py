import numpy as np
from helpers import *
from sklearn.metrics import mean_squared_error
from datetime import datetime

def get_mse(prediction, actual):
	prediction = prediction[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()].flatten()
	return mean_squared_error(prediction, actual)

class ctrmf():
	def __init__(self,
				 ratings_data,
				 recipe_feature,
				 n_hidden=100,
				 reg_term=0.0,
				 verbose=False):
		"""
		ratings_data : ((n_users, n_recipes) 2D array)
			Collection of recipe ratings for each user.
		recipe_feature : ((n_recipes, n_features) 2D array)
			Recipe-feature matrix, where each element is 1 if the recipe
			contains the corresponding feature, 0 otherwise.
		n_hidden : (Integer)
			Number of latent dimensions
		reg_term : (Double)
			Regularization term
		verbose : (Boolean)
			Prints helpful training progress messages if True
		"""
		self.ratings_data = ratings_data # users x recipes
		self.recipe_feature = recipe_feature # recipes x features
		self.n_hidden = n_hidden
		self.n_users, self.n_recipes = ratings_data.shape
		self.n_features = recipe_feature.shape[1]
		self.reg_term = reg_term
		self.ratings_users, self.ratings_recipes = self.ratings_data.nonzero()
		self.n_samples = len(self.ratings_users)
		self.verbose = verbose
		if self.verbose:
			print("Initializing ctrmf")
			print("==================")
			print(f"n_users: {self.n_users}")
			print(f"n_recipes: {self.n_recipes}")
			print(f"recipe_feature.shape: {recipe_feature.shape}")
			print(f"n_features: {self.n_features}")
			print(f"n_hidden: {self.n_hidden}")
			print(f"n_samples: {self.n_samples}")

	def train(self, n_iter=10, learning_rate=0.1):
		"""
		Reset all learned matrices and start training from scratch.

		Params
		======
		n_iter: (Integer)
			Number of iterations.
		learning_rate: (Double)
			Used to determine the rate of sgd calculations.
		"""
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
		"""
		Assumes all learned matrices are initialized and performs n_iter iterations
		of stochastic gradient descent.
		"""
		np.random.seed(11)
		for i in range(n_iter):
			if i % 10 == 0 and self.verbose:
				print('\tcurrent iteration: {}'.format(i))
			self.training_indices = np.arange(self.n_samples)
			np.random.shuffle(self.training_indices)
			self.sgd()

	def sgd(self):
		"""
		Perform stochastic gradient descent
		"""
		ctr = 0
		start_time = datetime.now()
		time_ctr = datetime.now()
		for index in self.training_indices:
			ctr += 1
			u = self.ratings_users[index]
			r = self.ratings_recipes[index]
			if self.verbose and ctr % 100 == 0:
				new_time = datetime.now()
				print(f"Predicted {ctr} users in time {new_time - time_ctr}, total time {new_time - start_time}")
				time_ctr = new_time

			rs = self.ratings_data[u].nonzero()[0]
			predictions = self.predict_user(u, rs)
			eu = self.ratings_data[u] - predictions[u]
			self.hidden_feature += self.learning_rate * \
					sum([eu[x] * np.outer(self.user_hidden[u], self.recipe_feature[x]) for x in rs])
			#self.hidden_feature += self.learning_rate * \
			#	sum([self.error(u,x) * np.outer(self.user_hidden[u], self.recipe_feature[x]) for x in range(self.n_recipes)])
			self.recipe_bias[r] += self.learning_rate * (eu[r] - self.reg_term * self.recipe_bias[r])
			self.user_bias[u] += self.learning_rate * (eu[r] - self.reg_term * self.user_bias[u])
			self.user_hidden[u] += self.learning_rate * \
				sum([eu[x] * self.hidden_feature.dot(self.recipe_feature[x]) - self.reg_term * self.user_hidden[u] for x in rs])
	
	def error(self, u, r):
		return self.ratings_data[u, r] - self.predict(u, r)

	def predict_all(self):
		return [predict_user(u) for u in range(n_users)]

	def predict_user(self, u, rs=None):
		predictions = np.zeros(self.n_recipes)
		if rs is None:
			rs = range(self.n_recipes)
		for r in rs:
			predictions[r] = self.predict(u, r)
		return predictions

	def predict(self, u, r):
		prediction = self.global_bias + self.user_bias[u] + self.recipe_bias[r]
		prediction += self.user_hidden[u,:].dot(self.hidden_feature).dot(self.recipe_feature[r,:])
		return prediction

	def calculate_learning_curve(self, iter_array, test):
		"""

		"""
		iter_array.sort()
		self.train_mse = []
		self.test_mse = []
		iter_diff = 0
		for (i, n_iter) in enumerate(iter_array):
			if self.verbose:
				print("Iteration:", n_iter)
			if i == 0:
				self.train(n_iter)
			else:
				self.partial_train(n_iter - iter_diff)

			predictions = self.predict_all()

			self.train_mse += [get_mse(predictions, self.ratings_data)]
			self.test_mse += [get_mse(predictions, test)]
			if self.verbose:
				print("Train mse:", self.train_mse[-1])
				print("Test mse:", self.test_mse[-1])
			iter_diff = n_iter
		return train_mse, test_mse


R = get_recipe_feature_map().T
A = get_user_rating_matrix()
A_train, A_test = split_to_train_test(A, 0.2)
R_train = R[:32654]
print(f"R_train.shape: {R_train.shape}")

CTRMF = ctrmf(A_train, R_train, verbose=True, reg_term=0.01)

CTRMF.train(learning_rate=0.001)
