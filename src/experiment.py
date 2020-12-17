from helpers import *
from ctrmf import CTRMF
from explicitmf import ExplicitMF
import code
import sys
sys.path.append('../')
from Data.process_data import Bundle

non_ingredient_feature_num = 15
pulled_pork_recipe_id = '14731'
pork_roast_recipe_id = '39447'

# load experiment models
user_rating_matrix = get_experiment_user_rating_matrix()
recipe_feature_matrix = get_experiment_recipe_feature_map().T
# Get index maps
user_index, recipe_index = get_index_maps()
ingredient_to_index = ingredient_to_index()

user_one_idx = user_index['1']
user_two_idx = user_index['3']

pulled_pork_idx = recipe_index[pulled_pork_recipe_id]
pork_roast_idx = recipe_index[pork_roast_recipe_id]

user_one_tensor = torch.LongTensor([user_one_idx])
user_two_tensor = torch.LongTensor([user_two_idx])

# Init Tensors
pulled_pork_tensor = torch.LongTensor([pulled_pork_idx])
pork_roast_tensor = torch.LongTensor([pork_roast_idx])

# Find averages for recipes
pulled_pork_total = 0
pork_roast_total = 0
pulled_pork_ctr = 0
pork_roast_ctr = 0
for i in range(len(user_rating_matrix)):
    pulled_pork_total += user_rating_matrix[i][pulled_pork_idx]
    if user_rating_matrix[i][pulled_pork_idx] != 0:
        pulled_pork_ctr += 1
    pork_roast_total += user_rating_matrix[i][pork_roast_idx]
    if user_rating_matrix[i][pork_roast_idx] != 0:
        pork_roast_ctr += 1
print("")
print("Averages")
print(f"average pulled pork rating: {pulled_pork_total/pulled_pork_ctr}")
print(f"average pork roast rating: {pork_roast_total/pork_roast_ctr}")
print("")
print("MF experiment results")
# Get MF results
model = pickle.load(open('../models/explicitmodel.pkl', 'rb'))
mf_pred_1_pulled_pork = model.predict(user_one_idx, pulled_pork_idx)
mf_pred_1_pork_roast = model.predict(user_one_idx, pork_roast_idx)
mf_pred_2_pulled_pork  = model.predict(user_two_idx, pulled_pork_idx)
mf_pred_2_pork_roast = model.predict(user_two_idx, pork_roast_idx)

print(f"user 1 pulled pork rating: {mf_pred_1_pulled_pork}")
print(f"user 2 pulled pork rating {mf_pred_2_pulled_pork}")
print(f"user 1 pork roast rating: {mf_pred_2_pulled_pork}")
print(f"user 2 pork roast rating {mf_pred_2_pulled_pork}")
print("")
print("CTRMF experiment results")
# Get CTRMF results
model = CTRMF(user_rating_matrix, recipe_feature_matrix)
load_model(model, '../models/model.pkl', 'cpu')

pulled_pork_rating = model.forward(user_one_tensor, pulled_pork_tensor).item()
pork_roast_rating = model.forward(user_one_tensor, pork_roast_tensor).item()

user_two_pp_rating = model.forward(user_two_tensor, pulled_pork_tensor).item()
user_two_pr_rating = model.forward(user_two_tensor, pork_roast_tensor).item()

print(f"user 1 pulled pork rating: {pulled_pork_rating}")
print(f"user 2 pulled pork rating {user_two_pp_rating}")
print(f"user 1 pork roast rating: {pork_roast_rating}")
print(f"user 2 pork roast rating {user_two_pr_rating}")
