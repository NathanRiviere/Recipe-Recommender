import numpy as np
import pickle
import random
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def get_auc(pred, actual):
	classes = [1.0,2.0,3.0,4.0,5.0]
	pred = np.round(pred[actual.nonzero()].flatten())
	pred = label_binarize(pred, classes=classes)
	actual = actual[actual.nonzero()].flatten()
	actual = label_binarize(actual, classes=classes)
	n_classes = len(classes)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(actual[:, i], pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	return fpr, tpr, roc_auc

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

'''
# Pick out 20% of user ratings for our test set
def split_to_train_test(user_ratings):
    rating_count = np.count_nonzero(user_ratings)
    test_set_size = round(0.2 * rating_count)
    user_ratings_test = None
    test_count = 0
    print("Need " + str(test_set_size) + " ratings")
    while test_count <= test_set_size:
        print(str(test_count) + " ratings added to test set")
        test_column = np.copy(user_ratings[:,-1])
        test_count += np.count_nonzero(test_column)
        user_ratings = np.delete(user_ratings, -1, 1)

        if user_ratings_test is None:
            user_ratings_test = test_column
        else:
            user_ratings_test = np.column_stack((user_ratings_test, test_column))

    return user_ratings, user_ratings_test

'''

def split_to_train_test(user_ratings, test_percentage):
    total_columns = (user_ratings.shape)[1]
    test_column_count = round(test_percentage*total_columns)
    A_train = user_ratings[:, :-test_column_count]
    A_test = user_ratings[:, -test_column_count:]
    return A_train, A_test
