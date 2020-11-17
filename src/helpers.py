import sys
sys.path.append(".")
import numpy as np
import pickle
import random 

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
