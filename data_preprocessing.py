# data_preprocessing.py 

# 2020 T2 COMP9417 Group Project

# Group Member

"""

Shu Yang (z5172181)  
Yue Qi (z5219951)  
Tim Luo (z5115679) 
Yixiao Zhan (z5210796)

"""
 
# Import the relevant packages we'll need for this project.
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import json
import ast
import math
from collections import Counter, defaultdict
from ast import literal_eval
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import operator
import statistics
import calendar

# Read in training and test data  
train = pd.read_csv("./Data/train.csv")
test = pd.read_csv("./Data/test.csv")

# setting some global value
drop_list_Yue = []

###########################################################################
###   The following preprocess the "belongs_to_collection" attribute    ###
###########################################################################

# Relationship between revenue and collection.
# Movies with collection have more revenue
train['belongs_to_collection'].fillna(0, inplace=True)
draw_train = train
draw_train['newbelongs_to_collection'] = draw_train['belongs_to_collection'].apply(lambda x: literal_eval(x) if x != 0 else 0)
draw_train['name_collection'] = draw_train['newbelongs_to_collection'].apply(lambda x: 1 if x != 0 else 0)

train['belongs_to_collection'].fillna(0, inplace=True)
test['belongs_to_collection'].fillna(0, inplace=True)

def checkBelongs(target_list):
    # change the json into dic
    target_list['newbelongs_to_collection'] = target_list['belongs_to_collection'].apply(lambda x: literal_eval(x) if x != 0 else 0)

    # add one more collection name to store the collection name
    target_list['name_collection'] = target_list['newbelongs_to_collection'].apply(lambda x: x[0]['name'] if x != 0 else '0')
    # lable encoder
    le = LabelEncoder()
    le.fit(list(target_list['name_collection'].fillna('')))
    target_list['name_collection'] = le.transform(target_list['name_collection'].fillna('').astype(str))

checkBelongs(train)
checkBelongs(test)

# drop the old belongs_to_collection and newbelongs_to_collection
drop_list_Yue.append('newbelongs_to_collection')
drop_list_Yue.append('belongs_to_collection')
train.head()

###########################################################################
###         The following preprocess the "budget" attribute             ###
###########################################################################

# #### budget
# McKenzie (2012) believes that the relationship between budget and movie success is critical importance. So we choose to keep the 0 budget if the budget of this movie is not certain.

# checkout budget
train['budget'].fillna(0, inplace=True)
test['budget'].fillna(0, inplace=True)
def checkBudget(target_list):
    # change the json into dic
    target_list['budget_log'] = target_list['budget'].apply(lambda x: np.log(x) if x != 0 else 0)

checkBudget(train)
checkBudget(test)
# drop the old budget
drop_list_Yue.append('budget')

###########################################################################
###            The following preprocess the "genre" attribute           ###
###########################################################################

# checkout genres
train['genres'].fillna(0, inplace=True)
test['genres'].fillna(0, inplace=True)
# change the json into dic
train['new_genres'] = train['genres'].apply(lambda x: literal_eval(x) if x != 0 else 0)
test['new_genres'] = test['genres'].apply(lambda x: literal_eval(x) if x != 0 else 0)

train['new_genres'].apply(lambda x: len(x) if x != 0 else 0).value_counts()
# only 7 movies don't have the genres

# check each genres for each movie
def check_genre(target_list):
    list_of_genres = list(target_list['new_genres'].apply(lambda x: [i['name'] for i in x] if x != 0 else []).values)
    target_list['all_genres'] = target_list['new_genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != 0 else '')
    generes_list = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common()]
    for g in generes_list:
        target_list['genre_' + g] = target_list['all_genres'].apply(lambda x: 1 if g in x else 0)

check_genre(train)
check_genre(test)
drop_list_Yue.append('genres')
drop_list_Yue.append('new_genres')
drop_list_Yue.append('all_genres')

###########################################################################
###    The following preprocess the "home page" attribute               ###
###########################################################################

train['homepage'].fillna(0, inplace=True)
test['homepage'].fillna(0, inplace=True)

train['len_homepage'] = train['homepage'].apply(lambda x: 1 if x != 0 else 0)

# From the bar chart, the movie with homepage has more revenue.

# we transform homepage into boolean
train['len_homepage'] = train['homepage'].apply(lambda x: 1 if x != 0 else 0)
test['len_homepage'] = test['homepage'].apply(lambda x: 1 if x != 0 else 0)
# drop the origin data
drop_list_Yue.append('homepage')


###########################################################################
###         The following preprocess the "imdb id" attribute            ###
###########################################################################

# checkout imdb_id
train['imdb_id'].apply(lambda x: len(x) if x != 0 else 0).value_counts()

# Since every movie has its own imdb_id which is unique to each movie, we can ignore this attribute.

drop_list_Yue.append('imdb_id')

###########################################################################
###       The following preprocess the "original language" attribute    ###
###########################################################################

# checkout original_language
train['original_language'].apply(lambda x: x if x != 0 else 0).value_counts()

def checkLanguage(target_list):
    # lable encoder
    le = LabelEncoder()
    le.fit(list(target_list['original_language'].fillna('')))
    target_list['langua_type'] = le.transform(target_list['original_language'].fillna('').astype(str))

checkLanguage(train)
checkLanguage(test)
# drop the origin data
drop_list_Yue.append('original_language')

###########################################################################
###         The following preprocess the "original title" attribute     ###
###########################################################################

# check the whether title has been changed, check the title length
def checkTitle(target_list):
    # check whether equal
    target_list['original_title'].fillna('')
    target_list['ortitle_equal'] = 1
    target_list.loc[ target_list['original_title'] == target_list['title'] ,"ortitle_equal"] = 0 
    
    # title length account
    target_list['original_title_length'] = target_list['original_title'].str.len() 
    target_list['original_title_word'] = target_list['original_title'].str.split().str.len()

checkTitle(train)
checkTitle(test)
drop_list_Yue.append('original_language')


###########################################################################
###             The following preprocess the "overview" attribute       ###
###########################################################################

# check the overview length
def checkOverview(target_list):
    # title length account
    target_list['overview_length'] = target_list['overview'].str.len() 
    target_list['overview_word'] = target_list['overview'].str.split().str.len()

checkOverview(train)
checkOverview(test)
drop_list_Yue.append('overview')

###########################################################################
###    The following preprocess the "popularity" attribute              ###
###########################################################################

# Show the relationship between popularity and revenue
train['popularity'].fillna(0, inplace=True)
test['popularity'].fillna(0, inplace=True)

###########################################################################
###         The following preprocess the "poster path" attribute        ###
###########################################################################

# We ignore the 'poster_path'. Even the poster should have some effect on motivating cutomers to watch movie. The anaylsis for image is a tough work. Besides, our concentration is focusing on the data that we can manipulate. Hence, we drop the img.
drop_list_Yue.append('poster_path')


###########################################################################
###    The following preprocess the "production_companies" attribute    ###
###########################################################################

# checkout production_companies
# fill the NaN
train['production_companies'].fillna(0, inplace=True)
test['production_companies'].fillna(0, inplace=True)
# change the json into dic
train['newproduction_companies'] = train['production_companies'].apply(lambda x: literal_eval(x) if x != 0 else 0)

test['newproduction_companies'] = test['production_companies'].apply(lambda x: literal_eval(x) if x != 0 else 0)

train['newproduction_companies'].apply(lambda x: len(x) if x != 0 else 0).value_counts()


# check each production_companies for each movie
def check_produc_com(target_list):
    list_of_prodc = list(target_list['newproduction_companies'].apply(lambda x: [i['name'] for i in x] if x != 0 else []).values)
    target_list['all_prodc'] = target_list['newproduction_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != 0 else '')
    prodc_list = [m[0] for m in Counter([i for j in list_of_prodc for i in j]).most_common()]
    for g in prodc_list:
        target_list['prodc_' + g] = target_list['all_prodc'].apply(lambda x: 1 if g in x else 0)

check_produc_com(train)
check_produc_com(test)
drop_list_Yue.append('production_companies')
drop_list_Yue.append('newproduction_companies')
drop_list_Yue.append('all_prodc')

###########################################################################
###    The following preprocess the "production_countries" attribute    ###
###########################################################################

# Separate and extract country name from production_countries
train['production_countries_count'] = train['production_countries'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
train['all_produced_countries'] = train['production_countries'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Do the same to test set
test['production_countries_count'] = test['production_countries'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
test['all_produced_countries'] = test['production_countries'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Calculate mean revenue for these movies with respect to production countries count
revenue_means = {}
for value in train['production_countries_count'].unique():
    query = "production_countries_count=='" + str(value) + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("TOP 10 revenue by production countries count\n")
for v in counter.most_common(10):
    countries, mean_revenue = v
    print("Movie produced from " + str(countries) + " has mean revenue " + str(mean_revenue))

# Calculate mean revenue for these movies with respect to production countries
revenue_means = {}
for value in train['all_produced_countries'].unique():
    query = "all_produced_countries=='" + value + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("\nTOP 10 revenue by production countries\n")
for v in counter.most_common(10):
    countries, mean_revenue = v
    print("Movie produced from " + countries + " has mean revenue " + str(mean_revenue))

print("\n")


###########################################################################
###        The following preprocess the "release dates" attribute       ###
###########################################################################

# Break release into month, day and year and derive release_dayofweek from the date
# Add all these information to separate columns in train dataframe

# Split release month, day, year into sepearte columns
train[["release_month","release_day","release_year"]] = train["release_date"].str.split("/",expand=True).replace(np.nan, -1).astype(int)
# Change release year to 4 digits year
train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000
train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900
release_date = pd.to_datetime(train["release_date"])
# Add the derived release weekday name to train
train["release_dayofweek"] = release_date.dt.day_name()

# Do the same to test set
test[["release_month","release_day","release_year"]] = test["release_date"].str.split("/",expand=True).replace(np.nan, -1).astype(int)
# Change release year to 4 digits year
test.loc[ (test['release_year'] <= 19) & (test['release_year'] < 100), "release_year"] += 2000
test.loc[ (test['release_year'] > 19)  & (test['release_year'] < 100), "release_year"] += 1900
release_date = pd.to_datetime(test["release_date"])
# Add the derived release weekday name to test
test["release_dayofweek"] = release_date.dt.day_name()

# test release data contains null value, add mode to fill null value
test['release_dayofweek'].fillna('Friday',inplace=True)

# Calculate mean revenue for these movies with respect to release year
revenue_means = {}
for value in train['release_year'].unique():
    query = "release_year=='" + str(value) + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue 
counter = Counter(revenue_means)
print ("TOP 10 revenue by year\n")
for v in counter.most_common(10):
    year, mean_revenue = v
    print("Movie produced in " + str(year) + " has mean revenue " + str(mean_revenue))

# Calculate mean revenue for these movies with respect to release month
revenue_means = {}
for value in train['release_month'].unique():
    query = "release_month=='" + str(value) + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("\nTOP 10 revenue by month\n")
for v in counter.most_common(10):
    month, mean_revenue = v
    print("Movie produced in " + str(calendar.month_name[month]) + " has mean revenue " + str(mean_revenue))

# Calculate mean revenue for these movies with respect to release day of week
revenue_means = {}
for value in train['release_dayofweek'].unique():
    query = "release_dayofweek=='" + value + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("\nTOP 10 revenue by production countries count\n")
for v in counter.most_common(10):
    dayofweek, mean_revenue = v
    print("Movie produced in " + dayofweek + " has mean revenue " + str(mean_revenue))

print("\n")

# Encode the month and release_dayofweek to become boolean value (e.g. release_on_Wed, release_on_June)

# Encode release_dayofweek
encoder = preprocessing.LabelBinarizer()
release_dayofweek_transformed = encoder.fit_transform(train["release_dayofweek"])
release_dayofweek = pd.DataFrame(release_dayofweek_transformed)
# Add to train dataframe
train = pd.concat([train,release_dayofweek], axis=1)

# Rename column name for easier use
train.rename(columns={0 : "released_on_" + str(encoder.classes_[0]), 1 : "released_on_" + str(encoder.classes_[1]), 2 : "released_on_" + str(encoder.classes_[2]),3 : "released_on_" + str(encoder.classes_[3]),4 : "released_on_" + str(encoder.classes_[4]),5 : "released_on_" + str(encoder.classes_[5]),6 : "released_on_" + str(encoder.classes_[6])},inplace=True)

release_month_transformed = encoder.fit_transform(train["release_month"])
release_month = pd.DataFrame(release_month_transformed)
# Add to train dataframe
train = pd.concat([train,release_month], axis=1)

# Rename column name for easier use
train.rename(columns={0 : "released_on_" + "Jan", 1 : "released_on_" + "Feb", 2 : "released_on_" + "Mar",3 : "released_on_" + "Apr",4 : "released_on_" + "May",5 : "released_on_" + "Jun",6 : "released_on_" + "Jul",7 : "released_on_" + "Aug", 8 : "released_on_" + "Sep", 9 : "released_on_" + "Oct",10 : "released_on_" + "Nov",11 : "released_on_" + "Dec"},inplace=True)

# Do the same to test set
# Encode release_dayofweek
test_release_dayofweek_transformed = encoder.fit_transform(test["release_dayofweek"])
test_release_dayofweek = pd.DataFrame(test_release_dayofweek_transformed)
# Add to test dataframe
test = pd.concat([test,release_dayofweek], axis=1)

# Rename column name for easier use
test.rename(columns={0 : "released_on_" + str(encoder.classes_[0]), 1 : "released_on_" + str(encoder.classes_[1]), 2 : "released_on_" + str(encoder.classes_[2]),3 : "released_on_" + str(encoder.classes_[3]),4 : "released_on_" + str(encoder.classes_[4]),5 : "released_on_" + str(encoder.classes_[5]),6 : "released_on_" + str(encoder.classes_[6])},inplace=True)

release_month_transformed = encoder.fit_transform(test["release_month"])
release_month = pd.DataFrame(release_month_transformed)
# Add to test dataframe
test = pd.concat([test,release_month], axis=1)

# Rename column name for easier use
test.rename(columns={0 : "released_on_" + "Jan", 1 : "released_on_" + "Feb", 2 : "released_on_" + "Mar",3 : "released_on_" + "Apr",4 : "released_on_" + "May",5 : "released_on_" + "Jun",6 : "released_on_" + "Jul",7 : "released_on_" + "Aug", 8 : "released_on_" + "Sep", 9 : "released_on_" + "Oct",10 : "released_on_" + "Nov",11 : "released_on_" + "Dec"},inplace=True)


###########################################################################
###        The following preprocess the "runtime"" attribute            ###
###########################################################################

# Find median of runtime attribute
median = train["runtime"].median()
# Fill na with median runtime value
train["runtime"].fillna(median,inplace=True)

median = test["runtime"].median()
test["runtime"].fillna(median,inplace=True)

# Do the same to test set
median = test["runtime"].median()
test["runtime"].fillna(median,inplace=True)


###########################################################################
###      The following preprocess the "spoken languages" attribute      ###
###########################################################################

# Separate and extract language from spoken languages
train['spoken_languages_count'] = train['spoken_languages'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
train['all_spoken_languages'] = train['spoken_languages'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Do the same to test set
test['spoken_languages_count'] = test['spoken_languages'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
test['all_spoken_languages'] = test['spoken_languages'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Calculate mean revenue for these movies with respect to spoken languages count
revenue_means = {}
for value in train['spoken_languages_count'].unique():
    query = "spoken_languages_count=='" + str(value) + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("TOP 10 revenue by spoken languages count\n")
for v in counter.most_common(10):
    languages, mean_revenue = v
    print("Movie of "+ str(languages) + " spoken languages " + " has mean revenue " + str(mean_revenue))

# Calculate mean revenue for these movies with respect to spoken languages
revenue_means = {}
for value in train['all_spoken_languages'].unique():
    query = "all_spoken_languages=='" + value + "'"
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("\nTOP 10 revenue by spoken languages\n")
for v in counter.most_common(10):
    languages, mean_revenue = v
    print("Movie of spoken language " + languages + " has mean revenue " + str(mean_revenue))

print("\n")


###########################################################################
###           The following preprocess the "status" attribute           ###
###########################################################################

# Sum revenue of all released movie
print ("Released movie mean revenue = " + str(train.query("status=='Released'")['revenue'].mean()))
# Sum revenue of all rumoured movie
print ("Rumored movie mean revenue = " + str(train.query("status=='Rumored'")['revenue'].mean()))


# Clearly, released movies have a much higher mean revenue than rumored movie, 
# therefore, we will create a column isReleased to our train dataframe

train["is_released"] = np.where(train["status"]=="Released",1,0)
train.drop(columns=["status"],inplace=True)

# Do the same to test set
test["is_released"] = np.where(test["status"]=="Released",1,0)
test.drop(columns=["status"],inplace=True)

print("\n")

###########################################################################
###        The following preprocess the "keywords" attribute            ###
###########################################################################

# Separate and extract keyword from keywords
train['keywords_count'] = train['Keywords'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
train['all_keywords'] = train['Keywords'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Do the same to test set
test['keywords_count'] = test['Keywords'].apply(lambda x: len(ast.literal_eval(x)) if x == x else 0)
test['all_keywords'] = test['Keywords'].apply(lambda x: ",".join(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# Calculate mean revenue for these movies with resepct to keywords count
revenue_means = {}
for value in train['keywords_count'].unique():
    query = 'keywords_count=="' + str(value) + '"'
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("TOP 10 revenue by keywords count\n")
for v in counter.most_common(10):
    keywords, mean_revenue = v
    print("Movie with " + str(keywords) + " Keywords has mean revenue " + str(mean_revenue))

# Calculate mean revenue for these movies with respect to keywords
revenue_means = {}
for value in train['all_keywords'].unique():
    query = 'all_keywords=="' + value + '"'
    revenue_means[value] = round(train.query(query)['revenue'].mean(),2)

# Print the top 10 mean revenue
counter = Counter(revenue_means)
print ("\nTOP 10 revenue by keywords\n")
for v in counter.most_common(10):
    keywords, mean_revenue = v
    print("Movie of Keywords " + keywords + " has mean revenue " + str(mean_revenue))

print("\n")

###########################################################################
###           The following preprocess the "cast" attribute             ###
###########################################################################

# Replaces cast dict with list of cast
def cast_list(cell):
    if isinstance(cell, str):
        cell_contents = cell[1:-1].replace('{','').replace('}','').replace('"','').replace('\'','').split(',')
        cell = [x.replace(' name: ','') for x in cell_contents if "name" in x]
    return cell

train['cast'] = train['cast'].map(cast_list)

# Grab revenues
def get_cast_members_with_revenues(df):
    cast_members = {}
    for index, movie in df.iterrows():
        if isinstance(movie['cast'], list):
            for cast in movie['cast']:
                if cast in cast_members:
                    revenue = cast_members[cast] + movie['revenue']
                    cast_members[cast] = revenue
                else:
                    cast_members[cast] = movie['revenue']
    return cast_members

# Grab how many times cast was actually cast
def cast_frequency(df):
    cast_list= []
    for index, movie in df.iterrows():
        if isinstance(movie['cast'], list):
            for cast in movie['cast']:
                cast_list.append(cast)
    unique, counts = np.unique(cast_list, return_counts=True)
    return dict(zip(unique, counts))

# This will also be used to rank cast based on revenue
cast_freq = cast_frequency(train)
cast_with_revenues = get_cast_members_with_revenues(train)

# Sort by largest revenue earners to smallest
sorted_cast_by_revenue = dict(sorted(cast_with_revenues.items(), key=operator.itemgetter(1),reverse=True))
sorted_cast_by_revenue

# Convert to list
cast_rev = []
for key, value in sorted_cast_by_revenue.items():
    temp = [key, value]
    cast_rev.append(temp)

# Divide by frequency of occurence
for cast in cast_rev:
    cast[1] /= cast_freq[cast[0]]

# Convert back to dict
dict_cast = dict() 
for cast in cast_rev: 
    dict_cast[cast[0]] = cast[1]
dict_cast

# Add a new column to training data which includes the score of each movie based on cast
# This score is simply the sum of the values of each cast

def get_cast_score(cell):
    cast_score = 0
    if isinstance(cell, list):
        for cast in cell:
            if cast in dict_cast:
                cast_score += dict_cast[cast]
    return cast_score

train['cast_score'] = train['cast'].map(get_cast_score)

# Check. As you can see, the metric does an ok job, but it is not perfect.
check = train.sort_values(by=['cast_score'], ascending=False)
check[['original_title', 'revenue', 'budget', 'popularity', 'cast_score']].head(20)

# See how well this new metric works (higher is better)
train['cast_score'].corr(train['revenue'])
# 0.7155260245338373 (not bad, but not as good as budget which was ~0.75)

# Apply to test set
test['cast'] = test['cast'].map(cast_list)
test['cast_score'] = test['cast'].map(get_cast_score)

test[['original_title', 'budget', 'popularity', 'cast_score']].head()

###########################################################################
###           The following preprocess the "crew" attribute             ###
###########################################################################

# calculate list of revenue a crew is involved
crew_with_revenue = defaultdict(list)
for index, movie in train.iterrows():
   try:
       revenue = movie.get('revenue')
       crews = set()
       for crew in ast.literal_eval(movie.get('crew')):
           crews.add(crew['name'])
       for crew_name in crews:
           crew_with_revenue[crew_name].append(revenue)
   except ValueError:
       pass

# calculate the mean value of a crew's revenue
crew_with_mean_revenue = defaultdict(int)
for k in crew_with_revenue:
    crew_with_mean_revenue[k] = statistics.mean(crew_with_revenue[k])
    
# extract crew names from 'crew'
train['all_crew_members'] = train['crew'].apply(lambda x: list(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')
test['all_crew_members'] = test['crew'].apply(lambda x: list(sorted([i['name'] for i in ast.literal_eval(x)])) if x == x else '')

# add a new column 'crew_score' which is calculated by the mean revenue of all the crew members for each movie
# crew_score = 0 if the value for 'crew' is missing
train['crew_score'] = train['all_crew_members'].apply(lambda x: statistics.mean(list(crew_with_mean_revenue[cru] for cru in x) if (x == x and len(x) > 0) else [0]))
test['crew_score'] = test['all_crew_members'].apply(lambda x: statistics.mean(list(crew_with_mean_revenue[cru] for cru in x) if (x == x and len(x) > 0) else [0]))

print("Revenue crew score correlation: " + str(train['crew_score'].corr(train['revenue'])))

# Final Adjustments

def log(df):
    columns = df.columns.tolist()
    columns.remove('id')
    for col in columns:
        if df[col].values.max() > 1:
            df[col] = np.log(df[col])



numericals = train.columns[train.dtypes != object]
train_numericals = train[numericals]
num = test.columns[test.dtypes != object]
test_numericals = test[num]

np.seterr(divide = 'ignore')
pd.options.mode.chained_assignment = None
log(train_numericals)
log(test_numericals)

train_numericals.to_csv('final_train.csv', encoding = 'utf-8-sig', index=False)
test_numericals.to_csv('final_test.csv', encoding = 'utf-8-sig', index=False)

