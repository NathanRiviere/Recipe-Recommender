import numpy as np
from helpers import *
from datetime import datetime
import torch


class ctrmf_optimized(torch.nn.Module):
    def __init__(self,
                ratings_data,
                recipe_feature,
                n_hidden=80,
                reg_term=0.01,
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
        self.recipe_feature = recipe_feature
        self.n_hidden = n_hidden
        self.n_users, self.n_recipes = ratings_data.shape
        self.n_features = recipe_feature.shape[1]
        self.reg_term = reg_term
        self.verbose = verbose
        self.user_hidden = torch.nn.Embedding(
            self.n_users, self.n_hidden, sparse=True)
        self.hidden_feature = torch.nn.Embedding(
            self.n_features, self.n_hidden, sparse=True)

        if self.verbose:
            print("Initializing ctrmf")
            print("==================")
            print(f"n_users: {self.n_users}")
            print(f"n_recipes: {self.n_recipes}")
            print(f"recipe_feature.shape: {recipe_feature.shape}")
            print(f"n_features: {self.n_features}")
            print(f"n_hidden: {self.n_hidden}")
            print(f"user_hidden.shape: ({self.user_hidden.num_embeddings},{self.user_hidden.embedding_dim})")
            print(f"hidden_feature.shape: ({self.hidden_feature.num_embeddings},{self.hidden_feature.embedding_dim})")

    # Calculates the prediction
    def forward(self, user, recipe):
		prediction = self.global_bias + self.user_bias[u] + self.recipe_bias[r]
		prediction += self.user_hidden[u].dot(self.hidden_feature).dot(self.recipe_feature[r])
		return prediction


R = get_recipe_feature_map().T
A = get_user_rating_matrix()
A_train, A_test = split_to_train_test(A, 0.2)
R_train = R[:32654]

CTRMF = ctrmf_optimized(A_train, R_train, verbose=True)
