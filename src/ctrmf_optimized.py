import os
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
        self.ratings_users, self.ratings_recipes = self.ratings_data.nonzero()
        self.verbose = verbose
        self.user_hidden = torch.nn.Embedding(
            self.n_users, self.n_hidden)
        self.hidden_feature = torch.ones((self.n_hidden,self.n_features), dtype=torch.float32, requires_grad=True)
        self.user_bias = torch.ones((self.n_users,), dtype=torch.float32, requires_grad=True)
        self.recipe_bias = torch.ones((self.n_recipes,), dtype=torch.float32, requires_grad=True)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-6, weight_decay=1e-5)
        self.global_bias = torch.FloatTensor([np.mean(self.ratings_data[np.where(self.ratings_data != 0)])])

        if self.verbose:
            print("Initializing ctrmf")
            print("==================")
            print(f"n_users: {self.n_users}")
            print(f"n_recipes: {self.n_recipes}")
            print(f"recipe_feature.shape: {recipe_feature.shape}")
            print(f"n_features: {self.n_features}")
            print(f"n_hidden: {self.n_hidden}")
            print(f"user_hidden.shape: ({self.user_hidden.num_embeddings},{self.user_hidden.embedding_dim})")
            print(f"hidden_feature.shape: ({self.hidden_feature.size()})")
            print(f"user_bias: {self.user_bias.shape}")
            print('\n')

    def train(self, iterations):
        batch_size = 5000
        running_loss = 0
        count = 0
        print(f"\n beginning training... batch size={batch_size}")
        for i in range(iterations):
            shuffle = np.random.permutation(len(self.ratings_users))
            rows, cols = self.ratings_users[shuffle], self.ratings_recipes[shuffle]
            start_time = datetime.now()
            for row, col in zip(*(rows, cols)):
                count += 1
                self.optimizer.zero_grad()
                rating = torch.FloatTensor([self.ratings_data[row,col]])
                row = torch.LongTensor([row])
                col = torch.LongTensor([col])
                prediciton = self.forward(row, col)
                loss = self.loss_func(prediciton, rating)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if count % batch_size == 4999:
                    print(f'[iteration:{i}, batch:{(count+1)//batch_size}] batch loss: {running_loss/batch_size}')
                    running_loss = 0
            print(f"Iteration {i} took {datetime.now() - start_time}")
    # Calculates the prediction
    def forward(self, user, recipe):
        Rr = torch.FloatTensor(self.recipe_feature[recipe])
        res = torch.matmul(self.user_hidden(user),self.hidden_feature)
        res = torch.matmul(res, Rr.T)
        return res + self.user_bias[user] + self.recipe_bias[recipe] + self.global_bias

R = get_recipe_feature_map("C:\\School\\AI\\Project\\Recipe-Recommender\\Data\\generated-results\\Recipe-feature_map.npy").T
A = get_user_rating_matrix("C:\\School\\AI\\Project\\Recipe-Recommender\Data\\generated-results\\user-rating_matrix.npy")
A_train, A_test = split_to_train_test(A, 0.2)
R_train = R[:32654]
CTRMF = ctrmf_optimized(A_train, R_train, verbose=True)
CTRMF.train(30)
