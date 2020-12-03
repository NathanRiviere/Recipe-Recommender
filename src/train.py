import argparse
from datetime import datetime
import numpy as np
import os
import os.path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from helpers import *
from ctrmf import CTRMF


# Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
				help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=25,
				help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
				help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Set batch size
batch_size = 5000

# Use GPU (cuda) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create RatingDataset class to help with loading data
class RatingDataset(Dataset):
	def __init__(self, user_ratings):
		self.user_ratings = user_ratings
		self.row_indices, self.col_indices = user_ratings.nonzero()
	
	def __len__(self):
		return len(self.row_indices)
	
	def __getitem__(self, idx):
		row = self.row_indices[idx]
		col = self.col_indices[idx]
		rating = self.user_ratings[row, col]
		return torch.LongTensor([row]).to(device), \
				torch.LongTensor([col]).to(device), \
				torch.FloatTensor([rating]).to(device)

# Load the user-ratings and recipe-feature matrices
recipe_feature = get_recipe_feature_map(os.path.join(args['dataset'], 'generated-results', 'Recipe-feature_map.npy')).T
user_rating = get_recipe_feature_map(os.path.join(args['dataset'], 'generated-results', 'user-rating_matrix.npy'))

# Split data into test and training sets
ur_train, ur_test, rf_train, rf_test  = split_to_train_test(user_rating, recipe_feature, .2)

ur_train_indices = list(zip(ur_train.nonzero()))

# Create two loader objects for the training and test datasets
batch_size = 1000
train_dataloader = DataLoader(RatingDataset(ur_train), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(RatingDataset(ur_test), batch_size=batch_size, shuffle=True)

# Instantiate the model
model = CTRMF(
	user_rating,
	recipe_feature,
	device=device,
	verbose=True
).to(device)

# Use MSE as the loss function
loss_func = torch.nn.MSELoss()

# Use SGD to optimize the weights
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Find non-zero indices to train on
ratings_users, ratings_recipes = ur_train.nonzero()

train_mse = []
test_mse = []

print(f"beginning training... batch size={batch_size}")

# Train model:
for epoch in range(args['epochs']):
	train_loss_tot = 0
	train_count = 0
	for i, (row_batch, col_batch, rating_batch) in enumerate(train_dataloader):
		train_count += 1
		optimizer.zero_grad()

		# Predict rating and calculate loss
		prediction = model(row_batch.squeeze(), col_batch.squeeze())
		prediction = torch.diagonal(prediction)
		loss = loss_func(prediction, rating_batch.squeeze())

		# Backpropagate
		loss.backward()

		# Update the parameters
		optimizer.step()

		# Update loss total
		train_loss_tot += loss.item()

	test_loss_tot = 0
	test_count = 0
	with torch.no_grad():
		for i, (row_batch, col_batch, rating_batch) in enumerate(test_dataloader):
			test_count += 1

			# Predict rating and calculate loss
			prediction = model(row_batch.squeeze(), col_batch.squeeze())
			prediction = torch.diagonal(prediction)
			loss = loss_func(prediction, rating_batch.squeeze())

			# Update loss total
			test_loss_tot += loss.item()

	train_mse += [train_loss_tot / train_count]
	test_mse += [test_loss_tot / test_count]
	print('[epoch:{}] Train MSE: {}, Test MSE: {}'.format(
		epoch,
		train_mse[-1],
		test_mse[-1]
	))


print('Finished training!')
plot_learning_curve_2(list(range(args['epochs'])), train_mse, test_mse, args['plot'])
