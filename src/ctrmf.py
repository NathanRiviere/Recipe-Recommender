import numpy as np
import torch

class CTRMF(torch.nn.Module):
	def __init__(self,
				ratings_data,
				recipe_feature,
				n_hidden=80,
				reg_term=0.01,
				device=None,
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
		super().__init__()

		self.ratings_data = ratings_data
		self.recipe_feature = torch.FloatTensor(recipe_feature).to(device)
		self.n_hidden = n_hidden
		self.n_users, self.n_recipes = ratings_data.shape
		self.n_features = recipe_feature.shape[1]
		self.reg_term = reg_term
		self.verbose = verbose

		# Set learned matrices as Embeddings
		self.user_hidden = torch.nn.Embedding(
			self.n_users,
			self.n_hidden
		).to(device)
		self.hidden_feature = torch.nn.Embedding(
			self.n_hidden,
			self.n_features
		).to(device)
		self.user_biases = torch.nn.Embedding(self.n_users, 1).to(device)
		self.recipe_biases = torch.nn.Embedding(self.n_recipes, 1).to(device)

		# Initialize learned matrices
		torch.nn.init.xavier_uniform_(self.user_hidden.weight)
		torch.nn.init.xavier_uniform_(self.hidden_feature.weight)
		self.user_biases.weight.data.fill_(0.)
		self.recipe_biases.weight.data.fill_(0.)

		# Calculate the mean of the ratings data
		self.global_bias = torch.FloatTensor([np.mean(
			self.ratings_data[np.where(self.ratings_data != 0)]
		)]).to(device)

		if self.verbose:
			print("Initializing ctrmf")
			print("==================")
			print(f"n_users: {self.n_users}")
			print(f"n_recipes: {self.n_recipes}")
			print(f"recipe_feature.shape: {recipe_feature.shape}")
			print(f"n_features: {self.n_features}")
			print(f"n_hidden: {self.n_hidden}")
			print(f"user_hidden.shape: ({self.user_hidden.num_embeddings},{self.user_hidden.embedding_dim})")
			print(f"hidden_feature.shape: {self.hidden_feature.weight.shape}")
			print('\n')

	def forward(self, user, recipe):
		pred = torch.matmul(self.user_hidden(user), self.hidden_feature.weight)
		pred = torch.matmul(pred, self.recipe_feature[recipe].T)
		pred += self.user_biases(user) + self.recipe_biases(recipe) + self.global_bias
		return pred
