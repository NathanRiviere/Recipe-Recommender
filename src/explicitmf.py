from helpers import *
from datetime import datetime

class ExplicitMF():
	def __init__(self,
				 ratings,
				 n_hidden=40,
				 recipe_hidden_reg=0.0,
				 user_hidden_reg=0.0,
				 recipe_bias_reg=0.0,
				 user_bias_reg=0.0,
				 verbose=False):
		"""
		Train a matrix factorization model to predict empty
		entries in a matrix. The terminology assumes a
		ratings matrix which is ~ user x recipe

		Params
		======
		ratings : (ndarray)
			User x Item matrix with corresponding ratings

		n_hidden : (int)
			Number of latent hidden to use in matrix
			factorization model

		recipe_hidden_reg : (float)
			Regularization term for recipe latent hidden

		user_hidden_reg : (float)
			Regularization term for user latent hidden

		recipe_bias_reg : (float)
			Regularization term for recipe biases

		user_bias_reg : (float)
			Regularization term for user biases

		verbose : (bool)
			Whether or not to printout training progress
		"""

		self.ratings = ratings
		self.n_users, self.n_recipes = ratings.shape
		self.n_hidden = n_hidden
		self.recipe_hidden_reg = recipe_hidden_reg
		self.user_hidden_reg = user_hidden_reg
		self.recipe_bias_reg = recipe_bias_reg
		self.user_bias_reg = user_bias_reg
		self.sample_row, self.sample_col = self.ratings.nonzero()
		self.n_samples = len(self.sample_row)
		self.verbose = verbose

	def train(self, n_iter=10, learning_rate=0.1):
		""" Train model for n_iter iterations from scratch."""
		# initialize latent vectors
		self.user_hidden = np.random.normal(scale=1./self.n_hidden,\
										  size=(self.n_users, self.n_hidden))
		self.recipe_hidden = np.random.normal(scale=1./self.n_hidden,
										  size=(self.n_recipes, self.n_hidden))
		self.learning_rate = learning_rate
		self.user_bias = np.zeros(self.n_users)
		self.recipe_bias = np.zeros(self.n_recipes)
		self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
		self.partial_train(n_iter)


	def partial_train(self, n_iter):
		"""
		Train model for n_iter iterations. Can be
		called multiple times for further training.
		"""
		ctr = 1
		while ctr <= n_iter:
			if ctr % 10 == 0 and self.verbose:
				print('\tcurrent iteration: {}'.format(ctr))
			self.training_indices = np.arange(self.n_samples)
			np.random.shuffle(self.training_indices)
			self.sgd()
			ctr += 1

	def sgd(self):
		for idx in self.training_indices:
			u = self.sample_row[idx]
			i = self.sample_col[idx]
			prediction = self.predict(u, i)
			e = (self.ratings[u,i] - prediction) # error

			# Update biases
			self.user_bias[u] += self.learning_rate * \
								(e - self.user_bias_reg * self.user_bias[u])
			self.recipe_bias[i] += self.learning_rate * \
								(e - self.recipe_bias_reg * self.recipe_bias[i])

			#Update latent factors
			self.user_hidden[u, :] += self.learning_rate * \
									(e * self.recipe_hidden[i, :] - \
									 self.user_hidden_reg * self.user_hidden[u,:])
			self.recipe_hidden[i, :] += self.learning_rate * \
									(e * self.user_hidden[u, :] - \
									 self.recipe_hidden_reg * self.recipe_hidden[i,:])
	def predict(self, u, i):
		""" Single user and recipe prediction."""
		prediction = self.global_bias + self.user_bias[u] + self.recipe_bias[i]
		prediction += self.user_hidden[u, :].dot(self.recipe_hidden[i, :].T)
		return prediction

	def predict_all(self):
		""" Predict ratings for every user and recipe."""
		predictions = np.zeros((self.user_hidden.shape[0],
								self.recipe_hidden.shape[0]))
		for u in range(self.user_hidden.shape[0]):
			for i in range(self.recipe_hidden.shape[0]):
				predictions[u, i] = self.predict(u, i)

		return predictions

	def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
		"""
		Keep track of MSE as a function of training iterations.

		Params
		======
		iter_array : (list)
			List of numbers of iterations to train for each step of
			the learning curve. e.g. [1, 5, 10, 20]
		test : (2D ndarray)
			Testing dataset (assumed to be user x recipe).

		The function creates two new class attributes:

		train_mse : (list)
			Training data MSE values for each value of iter_array
		test_mse : (list)
			Test data MSE values for each value of iter_array
		"""
		iter_array.sort()
		self.train_mse =[]
		self.test_mse = []
		self.train_auc =[]
		self.test_auc = []
		iter_diff = 0
		for (i, n_iter) in enumerate(iter_array):
			if self.verbose:
				print('Iteration: {}'.format(n_iter))
			if i == 0:
				self.train(n_iter - iter_diff, learning_rate)
			else:
				self.partial_train(n_iter - iter_diff)

			time = datetime.now()
			predictions = self.predict_all()
			new_time = datetime.now()
			print(f"Predictions calculated in {new_time - time}")

			self.train_mse += [get_mse(predictions, self.ratings)]
			self.test_mse += [get_mse(predictions, test)]
			#self.train_auc += [get_auc(predictions, self.ratings)]
			#self.test_auc += [get_auc(predictions, test)]
			if self.verbose:
				print('Train mse: ' + str(self.train_mse[-1]))
				print('Test mse: ' + str(self.test_mse[-1]))
				#print('Train auc: ' + str(self.train_auc[-1]))
				#print('Test auc: ' + str(self.test_auc[-1]))
			iter_diff = n_iter

A = get_user_rating_matrix()
A = A[:1000]
train, test = split_to_train_test(A, 0.2)
print(train.shape)
MF_SGD = ExplicitMF(train, 40, verbose=True)
iter_array = list(range(1,3))
MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
plot_learning_curve(iter_array, MF_SGD)

