import csv

'''
TODO:
Choose Features
    a) Ingredients
        i) condense ingredients
    b) nutrition
        i) parse out important nutrition info
    c) Cooking directions
        i) parse out ubiquitous data (Ready in, prep time, # of instructions) 
        ii) discover possible features from the instructions
            1) get unique words and count to see what information is available across the data


FEATURES:
- Ingredients
- High in carbs
- Low in carbs
- High in protien
- Low in protien
... <other nutrition info>
- High prep time
- Low prep time
- High ready in time
- Low ready in time
- Lots of instructions
- Few instructions
- Lots of equipment
- Few equipment
- Lots of rare equipment
- Few rare equipment
- barbeque used
- Cusine (Identify by Cusine_dict[name])
'''


# ingredients format: <ingredient>^<ingredient>^...
def process_ingredients(ingredients):
    #print(ingredients)
    return

# directions format: {'directions': u'Prep\n<time>\nReady In\n<time>\n<instructions>}
    # time format: <value> <value type (s,m,h)> <value> <value type>....
    # instructions format: <instruction>\n<instruction>\n....
def process_cooking_directions(directions):
    print(directions)
    return 

# nutrition format:
def process_nutrition(nutritions):
    #print(nutritions)
    return

with open('core-data_recipe.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    column_headers = next(reader)
    print(column_headers)
    for row in reader:
        process_ingredients(row[column_headers.index('ingredients')])
        process_cooking_directions(row[column_headers.index('cooking_directions')])
        process_nutrition(row[column_headers.index('nutritions')])