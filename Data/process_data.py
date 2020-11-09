import csv
import re
import pickle
import operator
import numpy as np


class Bundle:
    def __init__(self, user_index_map, recipe_index_map):
        self.user_index_map = user_index_map
        self.recipe_index_map = recipe_index_map
    
    def serialize(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        

###########################################    CONSTANTS   #####################################################
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

ordered_feature_list["Ingredients"] = [
    'FewIngredients',
    'ManyIngredients',
]

ordered_feature_list["Nutrition"] = [
    'carbohydrates',
    'sugars',
    'calories',
    'fat',
    'protein',
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
feature_extractor['FewIngredients'] = lambda x: ingredient_num_extract(x, False)
feature_extractor['ManyIngredients'] = lambda x: ingredient_num_extract(x, False)
feature_extractor['carbohydrates'] = lambda recipe: nutrition_extract(recipe, 'carbohydrates')
feature_extractor['sugars'] = lambda recipe: nutrition_extract(recipe, 'sugars')
feature_extractor['calories'] = lambda recipe: nutrition_extract(recipe, 'calories')
feature_extractor['fat'] = lambda recipe: nutrition_extract(recipe, 'fat')
feature_extractor['protein'] = lambda recipe: nutrition_extract(recipe, 'protein')

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

def get_ingredient_map():
    pickle_file = open('ingred_to_index.pkl', 'rb')
    ingredients = pickle.load(pickle_file)
    pickle_file.close()
    return ingredients

def get_recipe_id_map():
    recipe_ids = {}
    with open('condensed-data_interaction.csv', 'r') as f:
        index = 0
        while(True):
            line = f.readline()
            if line == '':
                return recipe_ids
            recipe_id = (line.split(','))[1]
            if recipe_id not in recipe_ids:
                recipe_ids[recipe_id] = index
                index += 1

def get_user_index_map():
    user_id_index_map = {}
    index = 0
    with open('condensed-data_interaction.csv', 'r') as f:
        while(True):
            interaction = f.readline()
            if interaction == '':
                return user_id_index_map
            user_id = (interaction.split(','))[0]
            if user_id not in user_id_index_map:
                user_id_index_map[user_id] = index
                index += 1

# Globals
ingredient_map = get_ingredient_map()
recipe_index_map = get_recipe_id_map()
user_index_map = get_user_index_map()

bad_ingredients = [
    'white', 'water', 'fryer', 'sauce', 
    'topping', 'spread', 'fresh', 'fluid ounce) can', 
    'half and half', 'brown', 'red', 'old-fashioned',
    'optional', 'crisp', 'wet ingredients', 'decoration',
    'blue', 'sliced', 'chopped', 'jumbo', 'choice',
    'prepared', 'to serve', 'yellow', 'dry ingredients',
    'ice', 'cubed', 'large', 'cooked', 'refrigerated',
    'black', 'grated', 'skim', 'shredded', 'dark',
    'amber', 'half-and-half', 'green', 'orange',
    'light', 'frozen', 'melted', 'hot', 'cold',
    'optional', 'sweet'
]

###########################################   CONDENSE FUNCTIONS   ###########################################
def condense_ingredients():

    categories = {}

    food_data = open('generic-food.csv', encoding='utf-8', newline='')
    reader = csv.reader(food_data)
    columns = next(reader)
    for food in reader:
        categories[food[columns.index("FOOD NAME")].lower()] = 1 

    with open('core-data_recipe.csv', encoding='utf-8' , newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_headers = next(reader)
        try:
            for recipe in reader:
                ingredients = recipe[column_headers.index('ingredients')]
                ingredients = ingredients.lower()
                ingredients = ingredients.replace(':', '')
                ingredient_list = ingredients.split('^')

                for ingredient in ingredient_list:
                    has_category = False
                    if ingredient in categories:
                        has_category = True
                        categories[ingredient] += 1
                        print("Found category for " + ingredient)
                    else:
                        ingredient_words = ingredient.split(' ')
                        # Try to find a word in the ingredient that is a category
                        for word in ingredient_words:
                            if word in categories:
                                categories[word] += 1
                                print(ingredient + " will be condensed to " + word)
                                has_category = True
                                break
                    # didnt find any matches for ingredient
                    if not has_category:
                        if ingredient in bad_ingredients:
                            continue
                        print("Cannot find feature for " + ingredient)
                        print("Adding " + ingredient + " as a category")
                        categories[ingredient] = 1
        except Exception as e:
            print(e)
    
    # Remove all ingredients that appeared less than 4 times
    final_categories = {}
    index = 0
    for i in categories:
        if categories[i] > 4:
            # Instead of value being count, make it the index value for the ingredient in the feature matrix
            final_categories[i] = index
            index += 1
    
    output_file = open("ingred_to_index.pkl", 'wb')
    pickle.dump(final_categories, output_file)
    return


# Result: condensed-data_interaction.csv which contains 10000 users and 45000 recipes
def condense_users_and_recipes():
    with open('raw-data_interaction.csv', encoding='utf-8' ,newline='') as csvfile:
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
        with open('condensed-data_interaction.csv', 'w', newline='') as condensed_csvfile:
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
    if match is None:
        return False
    try: 
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
        print("In prep extract: " + str(e))
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
        print("in ready in extract: " + str(e))
        return False

def instruction_extract(directions, is_negative_feature):
    threshold = 0
    if is_negative_feature:
        threshold = 25
    else:
        threshold = 10
    pattern = re.compile(r'\..*?\}')
    match = re.search(pattern, directions)
    if match is None:
        print("match is none")
    try:
        instructions = (match[0].replace('.', '\\n')).split('\\n')
        if is_negative_feature:
            return len(instructions) >= threshold
        else:
            return len(instructions) <= threshold
    except Exception as e:
        print("in instruction extract " + str(e))
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
        print("in nutrition extract " + str(e))
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
        print("In ingredient num extract, " + str(e))
        return False

###########################################    HELPER FUNCTIONS   #####################################################

def extract_features(ingredients, cooking_directions, nutrition):
    try: 
        feature_map = []
        for feature in ordered_feature_list["Cooking"]:
            feature_map.append(int(feature_extractor[feature](cooking_directions)))
        for feature in ordered_feature_list["Ingredients"]:
            feature_map.append(int(feature_extractor[feature](ingredients)))
        for feature in ordered_feature_list["Nutrition"]:
            feature_map.append(int(feature_extractor[feature](nutrition)))
        
        ingredient_vec = [0]*len(ingredient_map)
        ingredients = ingredients.lower()
        ingredients = ingredients.replace(':', '')
        ingredient_list = ingredients.split('^')
        
        for ingredient in ingredient_list:
            if ingredient in ingredient_map:
                ingredient_vec[ingredient_map[ingredient]] = 1
            else:
                for word in ingredient:
                    if word in ingredient_map:
                        ingredient_vec[ingredient_map[word]] = 1
        
        feature_map += ingredient_vec
        return feature_map
    except Exception as e:
        print("in extract features, " + str(e))

def get_num_of_features():
    return len(ingredient_map) + len(ordered_feature_list["Cooking"]) + len(ordered_feature_list["Ingredients"]) + len(ordered_feature_list["Nutrition"])

def get_num_of_recipes():
    recipes = {}
    with open('condensed-data_interaction.csv', encoding='utf-8', newline='') as r:
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
        with open('condensed-data_interaction.csv', 'r') as f:
            while(True):
                interaction = f.readline()
                if interaction == '':
                    break
                fields = interaction.split(',')
                user_id = fields[0]
                recipe_id = fields[1]
                A[users[user_id], recipes[recipe_id]] = fields[2]
        save_file = open("A.npy", 'wb')
        np.save(save_file, A)
        save_file.close()
    except Exception as e:
        print("Error occured in create_A ", str(e))
    

def create_R():
    R_columns_num = get_num_of_recipes()
    R_rows_num = get_num_of_features()
    # R is the recipe-feature map
    R = np.zeros((R_rows_num, R_columns_num))
    # Make sure we know which column is which recipe in R
    R_id_index_map = {}
    index = 0
    with open('core-data_recipe.csv', encoding='utf-8' ,newline='') as recipe_data:
        reader = csv.reader(recipe_data)
        column_headers = next(reader)
        try: 
            for recipe in reader:
                if recipe[column_headers.index('recipe_id')] not in recipe_ids:
                    continue

                R_id_index_map[recipe[column_headers.index("recipe_id")]] = index
                recipe_map_vec = extract_features(recipe[column_headers.index('ingredients')], recipe[column_headers.index('cooking_directions')], recipe[column_headers.index('nutritions')])
                R[:, index] = np.asarray(recipe_map_vec)
                index += 1            
            r_file = open("R.npy", 'wb')
            np.save(r_file, R)
            r_file.close()
            print("Finished creating and saving R")
            print("Size of R is " + str(R.shape))
        except Exception as e:
            print("In create R, " + str(e))

###########################################    TEST FUNCTIONS   #####################################################

def run_tests():
    return 


###########################################    ENTRY POINTS   #####################################################

test = True
condense = False
create_matrices = False

if condense:
    condense_ingredients()
    condense_users_and_recipes()
if create_matrices:
    create_R()
    create_A()
    bundle = Bundle(user_index_map, recipe_index_map)
    bundle.serialize('index_maps.pickle')
if test:
    run_tests()


'''
How to load the dicts that map user/recipe id to the index of that user/recipe in a matrix
with open('index_maps.pickle', 'rb') as f:
    bundle = pickle.load(f)
    users = bundle.user_index_map
    recipes = bundle.recipe_index_map
'''