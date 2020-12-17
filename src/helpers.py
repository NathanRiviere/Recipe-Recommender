import numpy as np
import pickle
import random
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ctrmf
sns.set()

def get_experiment_user_rating_matrix(path='../Data/generated-results/test_user-rating_matrix.npy'):
	return np.load(path)

def get_experiment_recipe_feature_map(path="../Data/generated-results/test_recipe-feature_map.npy"):
    return np.load(path)

def get_user_rating_matrix(path='../Data/generated-results/user-rating_matrix.npy'):
	return np.load(path)

def get_recipe_feature_map(path='../Data/generated-results/Recipe-feature_map.npy'):
	return np.load(path)

def get_index_maps(path='../Data/generated-results/index_maps.pickle'):
	users = recipes = {}
	f = open(path, 'rb')
	bundle = pickle.load(f)
	# users dict: user_id -> index in user_rating_matrix
	users = bundle.user_index_map
	# recipes dict: recipe_id -> index in recipe_feature_map
	recipes = bundle.recipe_index_map
	return users, recipes

def ingredient_to_index(path='../Data/generated-results/ingredient_to_index.pickle'):
    f = open(path, 'rb')
    return pickle.load(f)

def get_mse(pred, actual):
	# Ignore nonzero terms.
	pred = pred[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()].flatten()
	return mean_squared_error(pred, actual)

def plot_learning_curve(iter_array, model):
	plt.title('MSE Training vs. Test')
	plt.plot(iter_array, model.train_mse, label='Training', linewidth=5)
	plt.plot(iter_array, model.test_mse, label='Test', linewidth=5)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('iterations', fontsize=30)
	plt.ylabel('MSE', fontsize=30)
	plt.legend(loc='best', fontsize=20)
	plt.show()

def plot_learning_curve_2(iter_array, train_mse, test_mse, plot):
	plt.title('MSE Training vs. Test')
	plt.plot(iter_array, train_mse, label='Training', linewidth=5)
	plt.plot(iter_array, test_mse, label='Test', linewidth=5)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('iterations', fontsize=30)
	plt.ylabel('MSE', fontsize=30)
	plt.legend(loc='best', fontsize=20)
	plt.show()
	plt.savefig(plot)

def split_to_train_test(user_ratings, recipe_feature, test_percentage):
	total_columns = (user_ratings.shape)[1]
	test_column_count = round(test_percentage*total_columns)
	ur_train = user_ratings[:, :-test_column_count]
	ur_test = user_ratings[:, -test_column_count:]
	rf_train = recipe_feature[:-test_column_count, :]
	rf_test = recipe_feature[-test_column_count:, :]
	return ur_train, ur_test, rf_train, rf_test

def load_model(model_container, filepath, device):
    model_container.load_state_dict(torch.load(filepath))

def save_model(model, filepath):
    f = open(filepath, 'wb')
    torch.save(model.state_dict(), filepath)
    f.close()
