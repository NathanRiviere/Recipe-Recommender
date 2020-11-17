import csv
import re
import pickle
import operator
import numpy as np
from itertools import combinations

DEBUG = False

debug_prefix = './Data/' if DEBUG else ''

class Bundle:
    def __init__(self, user_index_map, recipe_index_map):
        self.user_index_map = user_index_map
        self.recipe_index_map = recipe_index_map
        
    def serialize(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

class Logger:
    def __init__(self, filename):
        self._file = open(debug_prefix + 'output/' + filename, 'w')

    def log(self, arg):
        self._file.write(arg)
        self._file.write('\n')

###########################################    GLOBALS   #####################################################
ordered_feature_list = {}
ordered_feature_list["Cooking"] = [
    'LongPrepTime',
    'ShortPrepTime',
    'LongReadyInTime',
    'ShortReadyInTime',
    'ManyInstructions',
    'FewIntstructions',
    'ManyEquipment',
    'FewEquipment',
]

ordered_feature_list["Nutrition"] = [
    'carbohydrates',
    'sugars',
    'calories',
    'fat',
    'protein',
]

ordered_feature_list["Ingredients"] = [
    'FewIngredients',
    'ManyIngredients',
    'Ingredients'
]

feature_extractor = {}
feature_extractor['LongPrepTime'] = lambda x: prep_extract(x, True) 
feature_extractor['ShortPrepTime'] = lambda x: prep_extract(x, False)
feature_extractor['LongReadyInTime'] = lambda x: ready_in_extract(x, True)
feature_extractor['ShortReadyInTime'] = lambda x: ready_in_extract(x, False)
feature_extractor['ManyInstructions'] = lambda x: instruction_extract(x, True)
feature_extractor['FewIntstructions'] = lambda x: instruction_extract(x, False)
feature_extractor['ManyEquipment'] = lambda x: equipment_extract(x, True)
feature_extractor['FewEquipment'] = lambda x: equipment_extract(x, False)
feature_extractor['carbohydrates'] = lambda recipe: nutrition_extract(recipe, 'carbohydrates')
feature_extractor['sugars'] = lambda recipe: nutrition_extract(recipe, 'sugars')
feature_extractor['calories'] = lambda recipe: nutrition_extract(recipe, 'calories')
feature_extractor['fat'] = lambda recipe: nutrition_extract(recipe, 'fat')
feature_extractor['protein'] = lambda recipe: nutrition_extract(recipe, 'protein')
feature_extractor['FewIngredients'] = lambda x: ingredient_num_extract(x, False)
feature_extractor['ManyIngredients'] = lambda x: ingredient_num_extract(x, True)
feature_extractor['Ingredients'] = lambda ingredients: ingredient_extract(ingredients)

equipment_list = [
    'apple cutter', 
    'baster',
    'pot',
    'pan',
    'wok',
    'torch',
    'bottle opener',
    'bread knife',
    'tray',
    'cleaver',
    'cutting board',
    'filet knife',
    'food mill',
    'grater',
    'strainer',
    'press',
    'ladle',
    'lemon squeezer',
    'measuring cup',
    'measuring spoon',
    'tenderiser',
    'thermometer',
    'nut cracker',
    'peeler',
    'pizza cutter',
    'scale',
    'tongs',
    'tong',
    'whisk',
    'zester',
    'microwave',
    'barbeque',
    'oven',
    'slow cooker',
    'rice cooker',
    'hot plate',
    'skillet',
    'saucepot',
]

def get_ingredient_count():
    pickle_file = open(debug_prefix + 'generated-results/ingredient_count.pickle', 'rb')
    ingredients = pickle.load(pickle_file)
    pickle_file.close()
    return ingredients    

def get_ingredient_map():
    pickle_file = open(debug_prefix + 'generated-results/ingredient_to_index.pickle', 'rb')
    ingredients = pickle.load(pickle_file)
    pickle_file.close()
    return ingredients

def get_recipe_id_map():
    recipe_ids = {}
    with open(debug_prefix + 'datasets/condensed-data_interaction.csv', 'r') as f:
        index = 0
        while(True):
            line = f.readline()
            if line == '':
                return recipe_ids
            recipe_id = (line.split(','))[1]
            if recipe_id not in recipe_ids:
                recipe_ids[recipe_id] = index
                index += 1
    x = len(recipe_ids)
    return recipe_ids

def get_user_index_map():
    user_id_index_map = {}
    index = 0
    with open(debug_prefix + 'datasets/condensed-data_interaction.csv', 'r') as f:
        while(True):
            interaction = f.readline()
            if interaction == '':
                return user_id_index_map
            user_id = (interaction.split(','))[0]
            if user_id not in user_id_index_map:
                user_id_index_map[user_id] = index
                index += 1

ignore_words = 'rinsed lightly thick pieces thin halved halves cubed cube very ripe wrapped unwrapped fine superfine new old trimmed inch inches ice green red blue orange yellow boiling stemmed frozen degrees degree warm cold temp topping diced ounce ounces fluid fluids thawed drained needed melted undrained halved prepared crumbled refridgerated canned mashed crushed smashed dried crushed grated flaked fresh shredded minced warm cold dry wet stale chop chopped fileted skinned touched grilled heated taste to and with for when where if into small medium large optional fluid ounce can such uncooked cooked ficed sliced beat beaten peeled pitted cut as for to ground toothpick toothpicks'.split(' ')

err_log = Logger("err.txt")

ingredient_map = {}
###########################################   CONDENSE FUNCTIONS   ###########################################

def get_all_sub_ingredients(ingredient_arr):
    # Remove words we want to ignore
    ingredient_arr = [i for i in ingredient_arr if i not in ignore_words and not i.isdigit() and len(i) > 2]
    # Remove 
    word_count = len(ingredient_arr)
    for i in range(2, len(ingredient_arr)+1):
        ingredient_arr += [' '.join(ingredient_arr[start:start+i]) for start in range(word_count-i + 1)]
    ingredient_arr.reverse()
    return ingredient_arr

def condense_ingredients(data_path):
    categories = {}
    food_data = open(debug_prefix+'datasets/generic-food.csv', encoding='utf-8', newline='')
    food_reader = csv.reader(food_data)
    for food in food_reader:
        categories[food[0].lower()] = 1 

    with open(data_path, encoding='utf-8' , newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_headers = next(reader)
        for recipe in reader:
            ingredients = recipe[column_headers.index('ingredients')].lower()
            non_letters = re.compile('[^a-zA-Z\\^]')
            ingredients = non_letters.sub(' ', ingredients).split('^')
            for ingredient in ingredients:
                ingredient_as_arr = ingredient.split(' ')
                is_one_word = len(ingredient) == 1
                sub_ingredients = get_all_sub_ingredients(ingredient_as_arr)
                best_match = None
                # Matches longest substring of ingredient that is a category
                # Creates categories for all substrings longer than the longest match (rare matches are removed at the end)
                for sub in sub_ingredients:
                    word_count = len(sub.split(' '))
                    if best_match is not None and word_count < len(best_match.split(' ')):
                        break
                    if sub in categories:
                        best_match = best_match if best_match is not None and categories[best_match] > categories[sub] else sub
                        continue
                    categories[sub] = 1
                
                if best_match is not None:
                    categories[best_match] += 1

    # Remove rare ingredients
    categories = {k:v for (k,v) in categories.items() if v > 50}
    ingredient_count = categories.copy()

    index = 0
    for ing in categories:
        categories[ing] = index
        index += 1

    category_log = Logger("food_categories.txt")
    for ingredient in categories: 
        category_log.log(ingredient)

    output_file = open(debug_prefix+"generated-results/ingredient_to_index.pickle", 'wb')
    pickle.dump(categories, output_file)
    ingredient_map = categories

    ing_count = open(debug_prefix+'generated-results/ingredient_count.pickle', 'wb')
    pickle.dump(ingredient_count, ing_count)
    return

# Result: condensed-data_interaction.csv which contains 10000 users and 45000 recipes
def condense_users_and_recipes():
    with open(debug_prefix+'datasets/raw-data_interaction.csv', encoding='utf-8' ,newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_headers = next(reader)
        
        user_rating_amt = {}
        for interaction in reader:
            user_id = interaction[column_headers.index('user_id')]
            if user_id in user_rating_amt:
                user_rating_amt[user_id] += 1
            else:
                user_rating_amt[user_id] = 1
        
        # Gets 10000 most active users
        user_interaction_count = sorted(user_rating_amt.items(), key=lambda x: x[1] ,reverse=True)[:10000]
        most_active_users = [user[0] for user in user_interaction_count]

        csvfile.seek(0)
        reader = csv.reader(csvfile)
        column_headers = next(reader)

        # Write all user-recipe interactions for every recipe a user in the top 10000 most active has rated
        with open(debug_prefix + 'datasets/condensed-data_interaction.csv', 'w', newline='') as condensed_csvfile:
            count = 0
            for interaction in reader:
                if count % 500 == 0:
                    print("amount of recipes processed=" + str(count))
                user_id = interaction[column_headers.index('user_id')]
                if user_id in most_active_users:
                    condensed_csvfile.write(','.join(interaction))
                count += 1
    return

###########################################   EXTRACTION FUNCTIONS   ###########################################

def prep_extract(directions, is_negative_feature):
    pattern = re.compile(r'Prep\\n\d.*?\\n')
    match = re.search(pattern, directions)
    prep_time = 0
    threshold = 0
    minute_conversion = 1
    # We consider 60 minutes a long time for prep and 15 or less to be short
    if is_negative_feature:
        threshold = 60
    else:
        threshold = 15
    try: 
        if match is None:
            return False
        time = match[0][6:-2].split(' ')
        for i in range(len(time)-1, -1, -1):
            item = time[i]
            if item == 'h' or item == 'hours':
                minute_conversion = 60
            elif item == 'm' or item == 'minutes':
                minute_conversion = 1
            elif item == 'd' or item == 'days':
                minute_conversion = 1440
            else:
                prep_time += int(item) * minute_conversion
        if is_negative_feature:
            return prep_time >= threshold
        else:
            return prep_time <= threshold

    except Exception as e:
        # Some recipes dont have prep time info. Default to false in this case.
        err_log.log(str(e))
        return False

def ready_in_extract(directions, is_negative_feature):
    pattern = re.compile(r'Ready In\\n\d.*?\\n')
    match = re.search(pattern, directions)
    ready_in_time = 0
    threshold = 0
    minute_conversion = 1
    if is_negative_feature:
        threshold = 90
    else:
        threshold = 30
    if match is None:
        return False
    try:
        time = match[0][10:-2].split(' ')
        for i in range(len(time)-1, -1, -1):
            item = time[i]
            if item == 'h' or item == 'hours':
                minute_conversion = 60
            elif item == 'm' or item == 'minutes':
                minute_conversion = 1
            elif item == 'd' or item == 'days':
                minute_conversion = 1440
            else:
                ready_in_time += int(item) * minute_conversion
        if is_negative_feature:
            return ready_in_time >= threshold
        else:
            return ready_in_time <= threshold
    
    except Exception as e:
        err_log.log(str(e))
        return False

def instruction_extract(directions, is_negative_feature):
    threshold = 0
    if is_negative_feature:
        threshold = 16
    else:
        threshold = 8
    pattern = re.compile(r'\..*?\}')
    match = re.search(pattern, directions)
    if match is None:
        pattern = re.compile(r'm\\n.*')
        match = re.search(pattern, directions)
    try:
        instructions = (match[0].replace('\\n', '')).split('.')
        instructions = [x for x in instructions if x != '' and x != "'}"]
        if is_negative_feature:
            return len(instructions) >= threshold
        else:
            return len(instructions) <= threshold
    except Exception as e:
        err_log.log(str(e))
        return False

def equipment_extract(directions, is_negative_feature):
    equipment_count = 0
    threshold = 0

    if is_negative_feature:
        threshold = 10
    else:
        threshold = 5

    directions = directions.lower()

    for equipment in equipment_list:
        if equipment in directions:
            equipment_count += 1

    if is_negative_feature:
        return equipment_count >= threshold
    else:
        return equipment_count <= threshold

# We assume if a meal has more than 25% of the recommended intake, it is considered high in that nutrient
def nutrition_extract(recipe, nutrient):
    try:
        pattern = re.compile(r'' + nutrient + r'.*?\}')
        full_nutrient_info = re.search(pattern, recipe)
        daily_val_pattern = re.compile(r'percentDailyValue.*?\,')
        daily_value_field = re.search(daily_val_pattern, full_nutrient_info[0])
        extract_num_pattern = re.compile(r'\d+')
        value = re.search(extract_num_pattern, daily_value_field[0])
        if value is None:
            return False
        return int(value[0]) >= 25
    except Exception as e:
        err_log.log(str(e))
        return False

def ingredient_num_extract(ingredients, is_negative_feature):
    threshold = 0
    if is_negative_feature:
        threshold = 12
    else:
        threshold = 6
    try:
        num_of_ingredients = len(ingredients.split('^'))
        if is_negative_feature:
            return num_of_ingredients >= threshold
        else:
            return num_of_ingredients <= threshold
    except Exception as e:
        err_log.log(str(e))
        return False

def better_match(sub_ingredient, possible_match, best_match, match):
    if sub_ingredient in get_all_sub_ingredients(possible_match.split(' ')):
        # curr_sub is substring of possible match
        if best_match is None:
            return True
        if len(sub_ingredient) > len(best_match):
            # Always try and match the longer sub ingredient
            return True
        if len(sub_ingredient) < len(best_match):
            return False
        if ingredient_count[possible_match] > ingredient_count[match]:
            # Match more popular
            return True
        else:
            return False
    else:
        # ingredient is not a sub ingredient of possible match
        return False

def ingredient_extract(ingredients):
    ingredient_vec = [0]*len(ingredient_map)
    ingredients = ingredients.lower()
    non_letters = re.compile('[^a-zA-Z\\^]')
    ingredients = non_letters.sub(' ', ingredients).split('^')
    for ingredient in ingredients:
        ingredient_substrings = get_all_sub_ingredients(ingredient.split(' '))
        best_match = None
        match = None    
        ingredient_list = ingredient_map.keys()
        for sub in ingredient_substrings:
            # No more matches of same size as best match
            if best_match is not None and len(sub.split(' ')) < len(best_match.split(' ')):
                break
            
            if sub in ingredient_list:
                # Full substring is an ingredient in our list
                best_match = sub
                match = sub
                break

            for possible_match in ingredient_list:
                # Find the most popular ingredient that has sub as a substring
                if better_match(sub, possible_match, best_match, match):
                    best_match = sub
                    match = possible_match
                              
        if best_match is None:
            err_log.log("Failed to match " + ingredient)
            continue
        x = ingredient_map[match]
        ingredient_vec[ingredient_map[match]] = 1
    return ingredient_vec
            
###########################################    HELPER FUNCTIONS   #####################################################

def extract_features(ingredients, cooking_directions, nutrition):
    try: 
        feature_map = []
        for feature in ordered_feature_list["Cooking"]:
            feature_map.append(int(feature_extractor[feature](cooking_directions)))
        for feature in ordered_feature_list["Nutrition"]:
            feature_map.append(int(feature_extractor[feature](nutrition)))
        for feature in ordered_feature_list["Ingredients"]:
            if feature == 'Ingredients':
                feature_map += feature_extractor[feature](ingredients)
            else:
                feature_map.append(int(feature_extractor[feature](ingredients)))
        return feature_map
    
    except Exception as e:
        print(e)
        err_log.log(str(e))

def get_num_of_features():
    return len(ingredient_map) + len(ordered_feature_list["Cooking"]) + len(ordered_feature_list["Ingredients"]) + len(ordered_feature_list["Nutrition"]) - 1

def get_num_of_recipes():
    recipes = {}
    with open(debug_prefix + 'datasets/condensed-data_interaction.csv', encoding='utf-8', newline='') as r:
        while(True):
            line = r.readline()
            if line == '':
                break
            recipe_id = (line.split(','))[1]
            if recipe_id not in recipes:
                recipes[recipe_id] = 1
    return len(recipes)


###########################################    MAIN FUNCTIONS   #####################################################
def create_A():
    try:
        users = user_index_map
        recipes = recipe_index_map
        A_row_num = len(users)
        A_column_num = len(recipes)
        A = np.zeros((A_row_num, A_column_num))
        with open(debug_prefix+'datasets/condensed-data_interaction.csv', 'r') as f:
            while(True):
                interaction = f.readline()
                if interaction == '':
                    break
                fields = interaction.split(',')
                user_id = fields[0]
                recipe_id = fields[1]
                A[users[user_id], recipes[recipe_id]] = fields[2]
        save_file = open(debug_prefix+"generated-results/user-rating_matrix.npy", 'wb')
        np.save(save_file, A)
        save_file.close()
        print("Finished generating user-rating matrix")
        print("A is " + str(A.shape))
    except Exception as e:
        err_log.log(str(e))

def create_R():
    R_columns_num = get_num_of_recipes()
    R_rows_num = get_num_of_features()
    # R is the recipe-feature map
    R = np.zeros((R_rows_num, R_columns_num))
    # Make sure we know which column is which recipe in R
    count = 0
    with open(debug_prefix+ 'datasets/core-data_recipe.csv', encoding='utf-8' ,newline='') as recipe_data:
        reader = csv.reader(recipe_data)
        column_headers = next(reader)
        try: 
            for recipe in reader:
                if count % 500 == 0:
                    print(str(count) + " recipes mapped")
                if recipe[column_headers.index('recipe_id')] not in recipe_index_map:
                    continue

                index = recipe_index_map[recipe[column_headers.index("recipe_id")]]
                ingredient_map_vec = extract_features(recipe[column_headers.index('ingredients')], recipe[column_headers.index('cooking_directions')], recipe[column_headers.index('nutritions')])
                R[:, index] = np.asarray(ingredient_map_vec)
                count += 1
            r_file = open(debug_prefix+"generated-results/Recipe-feature_map.npy", 'wb')
            np.save(r_file, R)
            r_file.close()
            print("Finished generating Recipe-feature map")
            print("Size of R is " + str(R.shape))
        except Exception as e:
            print(e)
            err_log.log(str(e))

###########################################    TEST FUNCTIONS   #####################################################

def run_tests():
    R = np.load(debug_prefix+'generated-results/Recipe-feature_map.npy')
    A = np.load(debug_prefix+'generated-results/user-rating_matrix.npy')
    with open(debug_prefix+'datasets/test_data.csv', 'r') as test_data:
        reader = csv.reader(test_data)
        column_headers = next(reader)
        recipe = next(reader)

        column_index = recipe_index_map[recipe[column_headers.index('recipe_id')]]
        expected = R[:,column_index]
        np.zeros((1,len(expected)))
        t_ingredients = [
            'sauerkraut', 'granny smith apples', 'apple cider', 
            'brown sugar','salt', 'garlic powder', 'black pepper',
            'boneless pork loin roast', 'caraway seeds', 'italian seasoning',
            'onion'

        ]

        non_ingredient_feature_vec = np.array([0,1,1,0,0,1,0,1,0,0,0,0,1,0,1])
        ingredient_feautre_vec = np.zeros(len(ingredient_map))

        for ingredient in t_ingredients:
            ingredient_feautre_vec[ingredient_map[ingredient]] = 1

        result = np.concatenate((non_ingredient_feature_vec, ingredient_feautre_vec))

        if not np.array_equal(result, expected):
            print("Test 1 failed")
            for i in range(len(result)):
                if result[i] != expected[i]:
                    print("Index " + str(i) + " is not equal. test_val = " + str(result[i]) + ", expected = " + str(expected[i]))
        else:
            print("Test 1 passed")

        recipe = next(reader)

        column_index = recipe_index_map[recipe[column_headers.index('recipe_id')]]
        expected = R[:,column_index]
        np.zeros((1,len(expected)))
        t_ingredients = [
            'potatoes', 'bacon', 'sauce', 'heavy whipping cream', 'butter', 'garlic', 
            'parmesan cheese', 'crust', 'water', 'honey', 'active yeast', 'vegetable oil',
            'all purpose flour', 'mozzarella cheese'
        ]    

        non_ingredient_feature_vec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        ingredient_feautre_vec = np.zeros(len(ingredient_map))

        for ingredient in t_ingredients:
            ingredient_feautre_vec[ingredient_map[ingredient]] = 1

        result = np.concatenate((non_ingredient_feature_vec, ingredient_feautre_vec))

        if not np.array_equal(result, expected):
            print("Test 2 failed")
            for i in range(len(result)):
                if result[i] != expected[i]:
                    print("Index " + str(i) + " is not equal. test_val = " + str(result[i]) + ", expected = " + str(expected[i]))
        else:
            print("Test 2 passed")

    return 

###########################################    ENTRY POINTS   #####################################################
condense = True
create_matrices = True
runtest = True
test_data = False
data = debug_prefix+'datasets/test_data.csv' if test_data else debug_prefix+'datasets/core-data_recipe.csv'

if condense:
    condense_ingredients(data)
    condense_users_and_recipes()
# Globals
recipe_index_map = get_recipe_id_map()
user_index_map = get_user_index_map()
ingredient_map = get_ingredient_map()
ingredient_count = get_ingredient_count()
if create_matrices:
    create_R()
    create_A()
    bundle = Bundle(user_index_map, recipe_index_map)
    bundle.serialize(debug_prefix+'generated-results/index_maps.pickle')
if runtest:
    run_tests()

'''
How to load the dicts that map user/recipe id to the index of that user/recipe in a matrix
with open('index_maps.pickle', 'rb') as f:
    bundle = pickle.load(f)
    users = bundle.user_index_map
    recipes = bundle.recipe_index_map
'''