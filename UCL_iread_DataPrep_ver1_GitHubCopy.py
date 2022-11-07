# -*- coding: utf-8 -*-
"""Created on Thu Jun 24 15:27:43 2021
@author: szr4
"""
##############################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:30:27 2021

@author: szr4
"""     
#import psycopg2
#con = psycopg2.connect(database="iReadAnalyticsDB", user="iread_srizvi", password="*********", host="***.***.***.***", port="****")
#print("Database opened successfully")
#
#if con:
#    cursor = con.cursor()
#    print("\querying db")
#    query = ("SELECT activity_id as activity_id FROM analytics.activities WHERE input_type='words'")
#    cursor.execute(query)
#    
#    for (activity_id) in cursor:
#        print("{}".format(activity_id))

########################################################
#import psycopg2
#import sys, os
#import numpy as np
#import pandas as pd
##import ireaddbcreds as creds
#import pandas.io.sql as psql
#from datetime import datetime
#from dask import dataframe as dd
#import sqlite3
#
#startTime = datetime.now()  # Let's see how long this baby takes!
### ****** LOAD PSQL DATABASE ***** ##
## Set up a connection to the postgres server.
#conn = psycopg2.connect(database="iReadAnalyticsDB", user="iread_srizvi", password="******", host="***.***.***.***", port="****")
#print("Database opened successfully")
## Create a cursor object
#cursor = conn.cursor()
## Select the columns we're interested in
#selection = """
#SELECT
#    analytics.game_logs.game_log_id, analytics.game_logs.end_state, analytics.questions.question_game_log_id, analytics.questions.question_id, analytics.events.event_id, analytics.events.event_timestamp, analytics.events.event_question_id,  analytics.events.event_type
#FROM
#    analytics.game_logs
#        JOIN analytics.questions ON analytics.questions.question_game_log_id=analytics.game_logs.game_log_id
#        JOIN analytics.events ON analytics.events.event_question_id=analytics.questions.question_id
#WHERE
#    end_state <> 'quit'
#ORDER BY
#    game_log_id, question_id, event_timestamp;
#"""  # MAYBE WE HAVE TO USE DASK OR ANOTHER DB INSTEAD OF DIRECTLY INTO PANDAS
#
#cursor.execute(selection)
#results = cursor.fetchall()
#print("It has taken this long:", datetime.now() - startTime, "to grab the DB")
#
## READ IT INTO A DATAFRAME
#df = pd.read_sql(selection, conn)
#print("It has taken this long:", datetime.now() - startTime, "to grab and load the DB")
#
##CLOSE CONNECTION FOR GOOD PRACTICE
#cursor.close()
#conn.close()
## MAKE THE GAME LOG ID AND THE EVENT ID THE ROW HEADERS
#df = df.set_index(['game_log_id', 'question_id', 'event_id'])
## print(df.shape, df.head(20))
## GAME END BEING EVENT NUMBER 2 MAKES SORTING DIFFICULT, SO WE'LL CHANGE IT TO 6
#df = df.replace({'event_type':2}, 6)
#print(df.shape, df.head())
## SORT THE DATAFRAME
#df.sort_values(by=['game_log_id', 'question_id', 'event_timestamp', 'event_type'], inplace=True)
#print(df.shape, df.head(25))
#print("It has taken this long:", datetime.now() - startTime, "to grab and load and sort the DB")
##############################################################################

# first get and set the working directory
from os import chdir, getcwd
getcwd()
chdir('C:\\***')
getcwd()
###############################################################################

#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from numpy.random import seed
from numpy.random import randn
from scipy.stats import kruskal
from sklearn.decomposition import PCA

#iris = datasets.load_iris()
reevaluations = pd.read_csv('C:\\***\\analytics.reevaluations.csv')
print(reevaluations.columns)
reevaluate_logs = pd.read_csv('C:\\***\\analytics.reevaluate_logs.csv')
print(reevaluate_logs.columns)
profiles = pd.read_csv('C:\\***\\analytics.profiles.csv')
print(profiles.columns)
student_users = pd.read_csv('C:\\***\\analytics.student_users.csv')
print(student_users.columns)
features = pd.read_csv('C:\\***\\analytics.features.csv')
print(features.columns)
classes = pd.read_csv('C:\\***\\analytics.classes.csv')
print(classes.columns)
activities = pd.read_csv('C:\\***\\analytics.activities.csv')
print(activities.columns)
games = pd.read_csv('C:\\***\\analytics.games.csv')
print(games.columns)
game_logs = pd.read_csv('***\\analytics.game_logs.csv')
print(game_logs.columns)
activity_types = pd.read_csv('C:\\***\\analytics.activity_types.csv')
print(activity_types.columns)
metadata = pd.read_csv('C:\\***\\analytics.metadata.csv')
print(metadata.columns)
metadata_navigo = pd.read_csv('C:\\***\\analytics.metadata_navigo.csv')
print(metadata_navigo.columns)
metadata_metadata_navigo = pd.merge(metadata, metadata_navigo, how="inner", left_on=["metadata_id"], right_on = ["metadata_navigo_id"]) # left, right, outer, inner
print(metadata_metadata_navigo.head)
print(metadata_metadata_navigo.columns)
#select only useful columns, i.e. studnet_id, number of games played and total game duration
metadata_metadata_navigo = metadata_metadata_navigo[["metadata_user","metadata_navigo_games_played","metadata_navigo_games_duration"]]
print(metadata_metadata_navigo.columns)


# to extract initial competence, only select roaws with counter = 0, here old _ competence will be the initial competence value
reevaluations_zerocounter = reevaluations[reevaluations["reeval_elem_counter"] == 0]
reevaluations_zerocounter = reevaluations_zerocounter[['reeval_elem_feature_id', 'reeval_elem_old_competence','reeval_elem_reevaluation_log_id']]

# merge to get feature related details 
reeval_init = pd.merge(reevaluations_zerocounter, reevaluate_logs, how="inner", left_on=["reeval_elem_reevaluation_log_id"], right_on = ["reevaluation_log_id"]) # left, right, outer, inner
print(reeval_init.columns)

# rename the old competence column as initial competence in order to avoid any confusion
reeval_init.rename(columns=
{
"reeval_elem_old_competence": "initial_competence",
"reeval_elem_feature_id": "feature_id"
}, inplace=True)
print(reeval_init.columns)
#only keep what is needed to merge further
reeval_init = reeval_init[['feature_id', 'initial_competence','reevaluation_profile_id']]
print(reeval_init.columns)

reeval_curr = pd.merge(reevaluations, reevaluate_logs, how="inner", left_on=["reeval_elem_reevaluation_log_id"], right_on = ["reevaluation_log_id"]) # left, right, outer, inner
print(reeval_curr.columns)
# rename the old competence column as initial competence in order to avoid any confusion
reeval_curr.rename(columns=
{
"reeval_elem_new_competence": "current_competence",
"reeval_elem_feature_id": "feature_id"
}, inplace=True)
print(reeval_curr.columns)

#only keep what is needed to merge further
reeval_curr = reeval_curr[['feature_id', 'current_competence','reevaluation_profile_id']]
print(reeval_curr.columns)

# now merge both initial and current using prohile id
reeval_init_current = pd.merge(reeval_init, reeval_curr, how="inner", left_on=["feature_id","reevaluation_profile_id"], right_on = ["feature_id","reevaluation_profile_id"]) # left, right, outer, inner
print(reeval_init_current.columns)

# now merge with profile to get learner profile details and their current competence
reeval_profile = pd.merge(reeval_init_current, profiles, how="inner", left_on=["reevaluation_profile_id"], right_on = ["profile_id"]) # left, right, outer, inner
print(reeval_profile.columns)
reeval_profile['profile_user_id'].nunique()

# Get profiles from Spanish EFL domail model: EFL_ES_dm whose model id =8 
# EN_dm= 1, EN_dys_dm = 2, GR_dm =3, GR_dys_dm = 4, ES_dm =5, DE_dm = 6, EFL_GR_dm = 7, EFL_ES_dm = 8, EFL_RO_dm = 9, EFL_SW_dm =10

reeval_profile_ES_dm = reeval_profile[reeval_profile["profile_model_id"] == 8]
reeval_profile_ES_dm['profile_user_id'].nunique() # profile year is not avaoilable in profile but its in the student_user table so get it from there
std_comp_ES = pd.merge(student_users, reeval_profile_ES_dm, how="inner", left_on=["student_id"], right_on = ["profile_user_id"]) # left, right, outer, inner
print(std_comp_ES.columns)
std_comp_ES.nunique() # in student country check for few outliers from Greec (one id), and Romania
# clean this further

# Get profiles from English domain model: EN_dm whose model id =1

reeval_profile_EN = reeval_profile[reeval_profile["profile_model_id"] == 1]
reeval_profile_EN['profile_user_id'].nunique()
# now remove all profile year that doesnt have a value i.e. they are [-]
reeval_profile_EN = reeval_profile_EN[reeval_profile_EN["profile_year"] != "-"]
reeval_profile_EN['profile_user_id'].nunique()

print(reeval_profile_EN.columns)

# now only keep columns that are useful and can be combined with the model table and other stuff
reeval_profile_EN_keep = reeval_profile_EN[["profile_user_id", "profile_year","feature_id","initial_competence","current_competence"]]
reeval_profile_EN_keep['profile_user_id'].nunique()
reeval_profile_EN_keep['profile_user_id'].size
#print(df.keys())
#Checking for missing value
reeval_profile_EN_keep.isnull().sum()

# now merge with student
std_comp_EN = pd.merge(student_users, reeval_profile_EN_keep, how="inner", left_on=["student_id"], right_on = ["profile_user_id"]) # left, right, outer, inner
print(std_comp_EN.columns)
std_comp_EN.isnull().sum() # where class id, or school id is null...will be removed when we merge it with the class data
# will have to sort out null in year_group though
std_comp_EN.nunique() # notice how many countries are there? in students' country column?
# removing outliers from the student_country column because here the country is UK
std_comp_EN = std_comp_EN[std_comp_EN['student_country'] == "United Kingdom"]
std_comp_EN.nunique()
std_comp_EN = std_comp_EN[std_comp_EN['student_year_group'].notnull()]
std_comp_EN.nunique()#
std_comp_EN = std_comp_EN[std_comp_EN['student_gender'].notnull()] # just to be on safe side when use this code with other domain models
std_comp_EN.nunique()#
std_comp_EN.isnull().sum()

# now merge with ling. features dataset
std_f_comp_EN = pd.merge(std_comp_EN, features, how="inner", left_on=["feature_id"], right_on = ["feature_id"]) # left, right, outer, inner
std_f_comp_EN.nunique()

std_f_classes_comp_EN = pd.merge(std_f_comp_EN, classes, how="inner", left_on=["student_class_id"], right_on = ["class_id"]) 
std_f_classes_comp_EN.nunique()
list(std_f_classes_comp_EN['class_school'].unique())
# remove the class school with names Test, UCL # 10 student_id were associated with either UCL or Test
std_f_classes_comp_clean_EN = std_f_classes_comp_EN[(std_f_classes_comp_EN['class_school'] != 'UCL') & (std_f_classes_comp_EN['class_school'] !='Test')] 
std_f_classes_comp_clean_EN.nunique()# # 233 features have been played by 381 students from 3 year groups (y1, y2, y3) accross 30 classes with 11 school names
list(std_f_classes_comp_clean_EN['class_school'].unique())
# total number of schools ?
len(std_f_classes_comp_clean_EN['class_school'].unique()) 
# number of rows genenrated by each school?
std_f_classes_comp_clean_EN['class_school'].value_counts(sort = True, normalize = True)*100
#how many students id from each school?
std_sch_count = std_f_classes_comp_clean_EN.groupby('class_school')['student_id'].nunique()
# percentage of students participated from each school
(std_sch_count/378)*100 # there are 378 unique studnet id 

std_f_c_clean_EN = std_f_classes_comp_clean_EN[["student_id", 
                                                "student_registered_time", 
                                                "student_year_group",
                                                "student_gender",
                                                "student_birth_date", 
                                                "feature_id",
                                                "initial_competence",
                                                "current_competence",
                                                "linguistic_level",
                                                "category",
                                                "type",
                                                "description",
                                                "hr_category",
                                                "feature_difficulty",
                                                "class_school",
                                                "class_season"
                                                ]]

# now merge with activities
std_f_c_act_clean_EN_without_type = pd.merge(std_f_c_clean_EN, activities, how="inner", left_on=["feature_id"], right_on = ["activity_feature_id"]) 
std_f_c_act_clean_EN_without_type.nunique()
###############################################################################
# now merge with activity types (accuracy, blending, automaticity)
std_f_c_act_clean_EN_no_meta = pd.merge(std_f_c_act_clean_EN_without_type, activity_types, how="inner", left_on=["activity_type"], right_on = ["activity_type_id"]) 
std_f_c_act_clean_EN_no_meta.nunique()
list(std_f_c_act_clean_EN_no_meta['activity_type_code'].unique())
###############################################################################

std_f_c_act_clean_EN = pd.merge(std_f_c_act_clean_EN_no_meta, metadata_metadata_navigo, how="inner", left_on=["student_id"], right_on = ["metadata_user"]) 
std_f_c_act_clean_EN.nunique()

###############################################################################
std_f_c_act_clean_game_EN = pd.merge(std_f_c_act_clean_EN, games, how="inner", left_on=["activity_game_id"], right_on = ["game_id"]) 
std_f_c_act_clean_game_EN.nunique()
std_f_c_act_clean_game_EN['competence_change'] = (std_f_c_act_clean_game_EN['current_competence']) - (std_f_c_act_clean_game_EN['initial_competence'])
std_f_c_act_clean_game_EN.nunique() # 381 students, who played 233 features accross 15 games
list(std_f_c_act_clean_game_EN['competence_change'].unique())
list(std_f_c_act_clean_game_EN['game_name'].unique())
std_f_c_act_clean_game_EN.columns

std_f_c_act_clean_game_EN['competence_change'].describe() # 50% is median remember
std_f_c_act_clean_game_EN['competence_change'].skew() # 50% is median remember
std_f_c_act_clean_game_EN['competence_change'].kurtosis() 
std_f_c_act_clean_game_EN['competence_change'].std()
sorted(list(std_f_c_act_clean_game_EN['competence_change'].unique()))
###############################################################################
########## if needs normalization do it like below ############################
###############################################################################
x = std_f_c_act_clean_game_EN['competence_change']
normalized_competence_change = ((x-x.min())/(x.max()-x.min())*10).round(decimals=0)
normalized_competence_change.describe()
normalized_competence_change.skew()
normalized_competence_change.kurtosis()
normalized_competence_change.std()
sorted(list(normalized_competence_change.unique()))

###############################################################################
std_f_c_act_clean_game_EN[["initial_competence","current_competence","competence_change"]].describe()
std_f_c_act_clean_game_EN[["metadata_navigo_games_played"]].describe()

df_f_EN = std_f_c_act_clean_game_EN
df_f_EN.isnull().sum() # No column has null values anymore except for types because some doesnt have a type
#save this data in a csv file for the record
#df_f_EN.to_csv('df_f_EN.csv',index=False)

###############################################################################
# gender count, year group count, etc #
###############################################################################
df_f_EN.columns
# once done....make function for the summary below
student_count = df_f_EN["student_id"].nunique() # 378
gender_count = df_f_EN.groupby(['student_gender'], as_index=False)['student_id'].nunique()
gender_count
gender_prc = ((gender_count/student_count)*100).round(1)
gender_prc
(df_f_EN['student_gender'].value_counts('student_id')*100).plot.bar()

y_group_count = df_f_EN.groupby(['student_year_group'], as_index=False)['student_id'].nunique()
y_group_count
y_group_prc = ((y_group_count/student_count)*100).round(1)
y_group_prc
(df_f_EN['student_year_group'].value_counts('student_id')*100).plot.bar()

###############################################################################
# aggregating comtence_change score (mean of it) to game, activity_type_code, linglevel, cat, etc #
###############################################################################
# taking mean scores in each of the activity_type_code (accuracy, blending, automaticity), grouped bu student_id
                                # once done, make a function for the following that takes a list for group by and returns the aggregated dataframe
df_f_EN_actype = df_f_EN.groupby(['student_id','student_gender','student_year_group','activity_type_code'], as_index=False)['competence_change'].mean()
print(df_f_EN_actype.head)
print(df_f_EN_actype.columns)

# taking mean scores in each of the games, grouped bu student_id
df_f_EN_game_metadata = df_f_EN.groupby(['student_id','student_gender','student_year_group','metadata_navigo_games_played','metadata_navigo_games_duration','game_name'], as_index=False)['competence_change'].mean()
print(df_f_EN_game_metadata.head)
print(df_f_EN_game_metadata.columns)

# taking mean scores in each of the games, grouped bu student_id
df_f_EN_game = df_f_EN.groupby(['student_id','student_gender','student_year_group','game_name'], as_index=False)['competence_change'].mean()
print(df_f_EN_game.head)
print(df_f_EN_game.columns)
# taking mean scores in each of the linguistic level, grouped bu student_id
df_f_EN_llevel = df_f_EN.groupby(['student_id','student_gender','student_year_group','linguistic_level'], as_index=False)['competence_change'].mean()
print(df_f_EN_llevel.head)
print(df_f_EN_llevel.columns)

df_f_EN_category = df_f_EN.groupby(['student_id','student_gender','student_year_group','category'], as_index=False)['competence_change'].mean()
print(df_f_EN_category.head)
print(df_f_EN_category.columns)

df_f_EN_llevel_category = df_f_EN.groupby(['student_id','student_gender','student_year_group','linguistic_level','category'], as_index=False)['competence_change'].mean()
print(df_f_EN_llevel_category.head)
print(df_f_EN_llevel_category.columns)

df_f_EN_llevel_cat_actype = df_f_EN.groupby(['student_id','student_gender','student_year_group','linguistic_level','category','activity_type_code'], as_index=False)['competence_change'].mean()
print(df_f_EN_llevel_cat_actype.head)
print(df_f_EN_llevel_cat_actype.columns)

df_f_EN_llevel_cat_actype_game = df_f_EN.groupby(['student_id','student_gender','student_year_group','linguistic_level','category','activity_type_code','game_name','activity_id'], as_index=False)['competence_change'].mean()
print(df_f_EN_llevel_cat_actype_game.head)
print(df_f_EN_llevel_cat_actype_game.columns)

##################   subset for games   #######################################
df_f_EN["game_name"].unique()
df_f_EN.head()
d1g_rrf = df_f_EN[df_f_EN['game_name'] == "cleomatchra"].reset_index()
d1g_rrf.head

g_rrf_gender = pd.DataFrame(d1g_rrf.groupby(['student_id','student_gender'],as_index=False)['student_gender','competence_change'].mean())
g_rrf_gender.head()
g_rrf_gender["student_id"].nunique() # how many students played the game? 365 (365/378 = 96.6%)

new_g_rrf_gender = g_rrf_gender.pivot_table(columns='student_gender', index='student_id', values='competence_change').rename_axis(None, axis=1)
new_g_rrf_gender
g_rrf_female = new_g_rrf_gender[new_g_rrf_gender['female'].isnull() == False]
g_rrf_female = g_rrf_female[["female"]]
g_rrf_female.head

g_rrf_male = new_g_rrf_gender[new_g_rrf_gender["male"].isnull() == False]
g_rrf_male = g_rrf_male[["male"]]
g_rrf_male

###############################################################################

#need a function to seprate competence change valuses data for one specific column and for a specific valuse (like "female" in "student_gender")
# this will give you mean competence change for each student_id for that respective column
def sep(df, col1, val1):
    a_1 = pd.DataFrame(df.groupby(["student_id",col1],as_index=False)[col1,"competence_change"].mean())
    a_2 = a_1.pivot_table(columns=col1, index="student_id", values="competence_change").rename_axis(None, axis=1)
    a_3 = a_2[a_2[val1].isnull() == False]
    a_4 = a_3[[val1]]
    return pd.DataFrame(a_4)
###############################################################################
# normal data subset can be done this way therefore no need to introduce two columns in the above function as you may not always need two column being filtered 
df_f_EN_llevel_cat_actype_game["game_name"].unique()
df_f_EN_llevel_cat_actype_game.columns
new_df = df_f_EN_llevel_cat_actype_game[df_f_EN_llevel_cat_actype_game['game_name'] == "cleomatchra"].reset_index()
new_df.head
###############################################################################    
overall_female = sep(df_f_EN_llevel_cat_actype_game, "student_gender","female")
overall_male = sep(df_f_EN_llevel_cat_actype_game, "student_gender","male")

overall_year1 = sep(df_f_EN,"student_year_group","Year_1")
overall_year2 = sep(df_f_EN,"student_year_group","Year_2")
overall_year3 = sep(df_f_EN,"student_year_group","Year_3")

overall_accuracy = sep(df_f_EN_llevel_cat_actype_game, "activity_type_code","accuracy")
overall_accuracy.median()
overall_blending = sep(df_f_EN_llevel_cat_actype_game, "activity_type_code","blending")
overall_blending.median()
overall_auto = sep(df_f_EN_llevel_cat_actype_game, "activity_type_code","automaticity")
overall_auto.median()
###############################################################################
df_f_EN_llevel_cat_actype_game["game_name"].unique()

overall_cleomat = sep(df_f_EN_llevel_cat_actype_game, "game_name","cleomatchra")
overall_cleomat.median()
overall_perpath = sep(df_f_EN_llevel_cat_actype_game, "game_name","perilous paths")
overall_perpath.median()
overall_rrf = sep(df_f_EN_llevel_cat_actype_game, "game_name","raft rapid fire")
overall_rrf.median()
overall_croc = sep(df_f_EN_llevel_cat_actype_game, "game_name","crocotiles")
overall_croc.median()
overall_rtr = sep(df_f_EN_llevel_cat_actype_game,"game_name","remove the runes")
overall_rtr.median()
overall_slice = sep(df_f_EN_llevel_cat_actype_game,"game_name","sliceophagus")
overall_slice.median()
overall_croc_timed = sep(df_f_EN_llevel_cat_actype_game,"game_name","crocotiles-timed")
overall_croc_timed.median()
overall_hearo = sep(df_f_EN_llevel_cat_actype_game,"game_name","hearoglyphs")
overall_hearo.median()
overall_wys = sep(df_f_EN_llevel_cat_actype_game,"game_name","watch your step!")
overall_wys.median()
overall_sah = sep(df_f_EN_llevel_cat_actype_game,"game_name","saheara")
overall_sah.median()
overall_bridg = sep(df_f_EN_llevel_cat_actype_game,"game_name","bridgyptian")
overall_bridg.median()
overall_pillerpush = sep(df_f_EN_llevel_cat_actype_game,"game_name","pillar pusher")
overall_pillerpush.median()
overall_cart = sep(df_f_EN_llevel_cat_actype_game,"game_name","cart-astrophe")
overall_cart.median()
overall_anubrick = sep(df_f_EN_llevel_cat_actype_game,"game_name","anubrick")
overall_anubrick.median()
overall_cogelisk = sep(df_f_EN_llevel_cat_actype_game,"game_name","cogelisk")
overall_cogelisk.median()

#Mdn for the games cart-astrophe = 3.2
## Mdn for the games cleomatchra,crocotiles,crocotiles-timed, sliceophagus   = 3.0
#Mdn for the games perilous paths,raft rapid fire   = 2.3
#Mdn for the games remove the runes, hearoglyphs, saheara, cogelisk = 2.6
#Mdn for the games saheara, anubrick = 2.8
#Mdn for the games watch your step! = 2.0
#Mdn for the games bridgyptian = 1.8
#Mdn for the games pillar pusher = 1.3
###############################################################################

df_f_EN_llevel_cat_actype_game["linguistic_level"].unique()

overall_Morphology = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Morphology")
overall_Morphology.median()
overall_Morphosyntax = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Morphosyntax")
overall_Morphosyntax.median()
overall_Orthography = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Orthography")
overall_Orthography.median()
overall_Phonology = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Phonology")
overall_Phonology.median()
overall_Word_recognition = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Word recognition")
overall_Word_recognition.median()
overall_Syntax = sep(df_f_EN_llevel_cat_actype_game,"linguistic_level","Syntax")
overall_Syntax.median()
# Mdn for Word recognition = 5.0
# Mdn for Orthography = 3.5
# Mdn for Morphology, Morphosyntax, Syntax  = 3.0,
# Mdn for Phonology = 1.8

# Kruskal-Wallis H-test
from scipy.stats import kruskal
# rename the independent samples
data1 = overall_Morphology# cleomat # overall_female #g_rrf_female
data2 = overall_Morphosyntax# perpath # overall_male#g_rrf_male
data3 = overall_Orthography#rrf
data4 = overall_Phonology#croc
data5 = overall_Word_recognition#rtr
data6 = overall_Syntax#slice
#data7 = overall_croc_timed
#data8 = overall_hearo
#data9 = overall_wys
#data10 = overall_sah
#data11 = overall_bridg
#data12 = overall_pillerpush
#data13 = overall_cart
#data14 = overall_anubrick
#data15 = overall_cogelisk

# compare samples
stat, p = kruskal(data1, data2, data3, data4, data5, data6)#, data7, data8, data9
#                  , data10, data11, data12, data13, data14, data15)
print('Statistics=%.3f, p=%.3f, df = (number of data groups)-1' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions. No difference. (fail to reject H0)')
else:
	print('Different distributions. To understand the differences, examine medians and then do follow-up (reject H0)')

###############################################################################
# find median time a student spent on playing navigo     
df_f_EN_game_metadata_played_n_duration = df_f_EN_game_metadata[["student_id","student_gender","student_year_group","metadata_navigo_games_played","metadata_navigo_games_duration","competence_change"]]
df_f_EN_game_metadata_played_n_duration

std_id_median_played_n_duration = pd.DataFrame(df_f_EN_game_metadata_played_n_duration.groupby(["student_id","student_gender","student_year_group","competence_change"],as_index=False)["metadata_navigo_games_played", "metadata_navigo_games_duration"].median())
std_id_median_played_n_duration["metadata_navigo_games_duration (minutes)"] = ((std_id_median_played_n_duration["metadata_navigo_games_duration"])/1000)/60
std_id_median_played_n_duration["duration_per_game (minutes)"] = ((std_id_median_played_n_duration["metadata_navigo_games_duration (minutes)"])/(std_id_median_played_n_duration["metadata_navigo_games_played"]))

std_id_median_played_n_duration.columns # This is median games played by students in one go ( in a session) and how much time they have spent on each game 

sns.pairplot(std_id_median_played_n_duration[["student_gender", "student_year_group", "competence_change","metadata_navigo_games_played","metadata_navigo_games_duration (minutes)","duration_per_game (minutes)"]], 
                 hue="student_gender", palette="Set2", diag_kind="kde", height=5.0, hue_order = ["male","female"])

###############################################################################
# association between games played, duration and competence change
df = std_id_median_played_n_duration[["student_gender", "student_year_group", "competence_change","metadata_navigo_games_played","metadata_navigo_games_duration (minutes)","duration_per_game (minutes)"]]
new_df = df # df[df['student_year_group'] == "Year_3"]

plt.figure(figsize=(15,10))
correlations = new_df.corr()
correlations
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);
            
###############################################################################

#new_df = df_f_EN_llevel_cat_actype_game[df_f_EN_llevel_cat_actype_game['game_name'] == "cleomatchra"].reset_index()
#df_f_EN.columns
#df_f_EN["linguistic_level"].unique()
#df_f_EN["category"].unique()

#new_df = df_f_EN
#new_df['duration_per_game'] = (new_df['metadata_navigo_games_duration']/new_df['metadata_navigo_games_played'])/1000
#new_df = new_df[new_df['linguistic_level'] == 'Phonology']#.reset_index()
#new_df = new_df[new_df['category'] == 'Syllabification']
#new_df = new_df[new_df['student_year_group'] == 'Year_3']

#new_df = new_df[["competence_change","metadata_navigo_games_played"]] # "duration_per_game"#metadata_navigo_games_played
#new_df
#new_df.corr(method= 'pearson')

#df = std_f_c_clean_EN[std_f_c_clean_EN["current_competence", "feature_difficulty"]] 

#sns.pairplot(new_df, hue="student_year_group", height=2.5)

###############################################################################
def drawboxplot(x_axis,y_axis,dataframe):
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
    sns.set_context("paper", font_scale=2) # font_scale=0.9
    sns.boxplot(x= x_axis,y= y_axis,data=dataframe)
    plt.xticks(rotation = 45)
    plt.savefig(x_axis+' vs '+y_axis, bbox_inches='tight')   

#plt.subplot(3, 1, 1) # (row, column, figure number in that row and column)
#plt.subplot(3, 1, 1) # (row, column, figure number in that row and column)
#fig.suptitle('distribution of {} accross {}.'.format(y_axis, x_axis))

drawboxplot("game_name","competence_change",df_f_EN_llevel_cat_actype_game)
drawboxplot("student_gender","competence_change",df_f_EN_llevel_cat_actype_game)
drawboxplot("student_year_group","competence_change",df_f_EN_llevel_cat_actype_game)

###############################################################################
df_f_EN_llevel_cat_actype_game['game_type'] = df_f_EN_llevel_cat_actype_game['game_name'] + " (" + df_f_EN_llevel_cat_actype_game['activity_type_code'] + ") "
df_f_EN_llevel_cat_actype_game.nunique()
student_id_only = df_f_EN_llevel_cat_actype_game["student_id"].unique()
#df_f_EN_llevel_cat_actype_game.to_csv('df_f_EN_llevel_cat_actype_game.csv',index=False)

#SD : Indicates general variability within your sample data. Descriptive in nature. Use to assess overall variation and to estimate percentiles of normally distributed data.
#% CI: Indicates a likely range of values for a population parameter. Use to intuitively assess the certainty of an estimate and to compare with benchmarks.
sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
plt.xticks(rotation=45, horizontalalignment='right')
sns.set_context("paper", font_scale=1.5) # font_scale=0.9

# how many students played a particular game?
std_count_for_game = df_f_EN_llevel_cat_actype_game[["student_id", "student_gender", "student_year_group", "game_type"]].drop_duplicates().reset_index()
howmany = std_count_for_game.groupby(['game_type'])['student_id'].nunique()
howmany

sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
plt.xticks(rotation=45, horizontalalignment='right')
sns.set_context("paper", font_scale=1.5) # font_scale=0.9
sns.countplot( x ="game_type", hue="student_year_group", hue_order = ["Year_1", "Year_2", "Year_3"],
              data = std_count_for_game)
plt.legend(loc='upper right', title='year group') # upper right, upper left,lower left,lower right,right,center left,center right,lower center,upper center,center

# how many students played a particular linguistic level?
std_count_for_llevel = df_f_EN_llevel_cat_actype_game[["student_id", "student_gender", "student_year_group", "linguistic_level","category"]].drop_duplicates().reset_index()
howmany = std_count_for_llevel.groupby(['linguistic_level', 'category'])['student_id'].nunique()
howmany[howmany>99]
(howmany/378)*100 # here total 378 students

sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
plt.xticks(rotation=45, horizontalalignment='right')
sns.set_context("paper", font_scale=1.5) # font_scale=0.9
sns.countplot( x ="linguistic_level", hue="student_year_group", hue_order = ["Year_1", "Year_2", "Year_3"],
              data = std_count_for_llevel)
plt.legend(loc='upper right', title='year group') # upper right, upper left,lower left,lower right,right,center left,center right,lower center,upper center,center

#kindstr, optional: The kind of plot to draw, corresponds to the name of a categorical axes-level plotting function. Options are: “strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, or “count”.
# box plot
sns.catplot(x="student_year_group", y="competence_change", 
            hue="student_gender", col = "linguistic_level", kind="box",palette="pastel", 
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data = df_f_EN_llevel_cat_actype_game)
# bar plots 
# bar plots are more informative here showing the gander gap as well as year_group differences accross the games 
sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
sns.set_context("paper", font_scale=2.2) # font_scale=0.9
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "linguistic_level", kind="bar",palette="pastel", ci = "sd",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"], 
            data = df_f_EN_llevel_cat_actype_game)

df_f_EN_llevel_cat_actype_game.columns
df_f_EN_llevel_cat_actype_game["game_name"].unique()
# for a particular game: data= df_f_EN_game.query("game_name == 'raft_rapid_fire'")

sns.catplot(x="student_year_group", y="competence_change", 
            row = "game_name", kind="bar",palette="pastel", ci ="sd", 
            order=["Year_1", "Year_2", "Year_3"], 
            hue="student_gender",hue_order= ["male", "female"], 
            data = df_f_EN_llevel_cat_actype_game)#box

# You may have already noticed that bar plots are more informative here showing the gander gap as well as year_group differences accross the linguistic level
# lets query on data and select an interesting chunck
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "category", kind="bar",palette="pastel",  ci = "sd",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data = df_f_EN_llevel_cat_actype_game.query("linguistic_level == 'Phonology'"))# Word recognition

# boxplot for category
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            row = "category", kind="box",palette="pastel", 
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data = df_f_EN_llevel_cat_actype_game)

sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            row = "category", kind="bar",palette="pastel", 
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data = df_f_EN_llevel_cat_actype_game)

sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            row = "category", kind="bar",palette="pastel",  ci = "sd",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data = df_f_EN_llevel_cat_actype_game.query("category == 'GPC'"))

sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
sns.set_context("paper", font_scale=1.9) # font_scale=0.9
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "game_type", kind="bar",palette="pastel",  ci = "sd",col_wrap =3,
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("category == 'Clusters'")) # Syllabification # Frequency
#plt.ylim(-10, 10)

sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "category", kind="bar",palette="pastel",  ci = "sd",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("linguistic_level == 'Word recognition'"))
#plt.ylim(-10, 10)

sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "activity_type_code", kind="bar", ci="sd",palette="pastel", 
            col_order=["accuracy", "blending", "automaticity"],#col_wrap=2,
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game)  # data= df_f_EN_actype.query("activity_type_code == 'automaticity'") # # accuracy, blending

sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "activity_type_code", kind="bar",palette="pastel", ci = "sd", # box
            col_order=["accuracy", "blending", "automaticity"],
            row = "category",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("linguistic_level == 'Word recognition'"))# Word recognition, # Phonology, ###category == 'Syllabification
 
df_f_EN_llevel_cat_actype_game["competence_change"].mean()

#ci="sd" means error bar showing 1 sd in the observations
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "activity_type_code", kind="bar",palette="pastel", ci = "sd", # box
            col_order=["accuracy", "blending", "automaticity"],
            row = "category",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("category == 'Syllabification'"))# "linguistic_level == 'Phonology'" ## Word recognition, # Phonology
 
### note: the most negative competence change (reaching -3 points in Male students from year 3) was noticed in Phonology, in category of Syllabification, played in accuracy games such as (perilous path, remove the runes, watch your step)  
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "game_name", kind="bar",palette="pastel", ci = "sd", # box
            #col_order=["accuracy", "blending", "automaticity"],
            #row = "category",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("category == 'Syllabification'"))# "category == 'Irregular GPCs'"# "category == 'Syllabification'"# "linguistic_level == 'Phonology'" ## Word recognition, # Phonology

#df["period"] = df["Year"] + df["quarter"]
check_game_name = df_f_EN_llevel_cat_actype_game[["activity_type_code", "game_name"]].drop_duplicates().reset_index()
check_game_name

#df_f_EN_llevel_cat_actype_game['game_type'] = df_f_EN_llevel_cat_actype_game['game_name'] + " (" + df_f_EN_llevel_cat_actype_game['activity_type_code'] + ") "

plt.figure(figsize=(25, 20)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
sns.set_context("paper", font_scale=1.2) # font_scale=0.9
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "game_type", kind="bar",palette="pastel", ci = "sd", # box
            #col_order=["accuracy", "blending", "automaticity"],
            row = "category",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("linguistic_level == 'Phonology'"))# "category == 'Irregular GPCs'"# "category == 'Syllabification'"# "linguistic_level == 'Phonology'" ## Word recognition, # Phonology

###############################################################################
plt.figure(figsize=(25, 20)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
sns.set_context("paper", font_scale=1.5) # font_scale=0.9
sns.catplot(x="student_year_group", y="competence_change", hue="student_gender", 
            col = "game_type", col_wrap=3, kind="bar",palette="pastel", ci = "sd", # point# box# col_wrap=3
            #row = "category",
            order=["Year_1", "Year_2", "Year_3"], hue_order= ["male", "female"],
            data= df_f_EN_llevel_cat_actype_game.query("category == 'Syllabification'"))# "category == 'Irregular GPCs'"# "category == 'Syllabification'"# "linguistic_level == 'Phonology'" ## Word recognition, # Phonology

###############################################################################
#######duration and games played ##############################################
###############################################################################

df_f_EN_game_metadata_played_n_duration.columns

sns.set_style("whitegrid")
plt.figure(figsize=(20, 10)) # For two column paper (3.1, 3). Each column is about 3.15 inch wide
plt.xticks(rotation=45, horizontalalignment='right')
sns.set_context("paper", font_scale=1.5) # font_scale=0.9
sns.countplot( x ="metadata_navigo_games_played", hue="student_gender", hue_order = ["male", "female"],#hue="student_year_group", hue_order = ["Year_1", "Year_2", "Year_3"],
              data = df_f_EN_game_metadata_played_n_duration)
plt.legend(loc='upper right', title='year group') # upper right, upper left,lower left,lower right,right,center left,center right,lower center,upper center,center

###############################################################################
#### current competence level #################################################

currComp_df_f_EN_llevel_cat_actype_game = df_f_EN.groupby(['student_id','student_gender','student_year_group','linguistic_level','category','activity_type_code','game_name'], as_index=False)['current_competence'].mean()
print(currComp_df_f_EN_llevel_cat_actype_game.head)
print(currComp_df_f_EN_llevel_cat_actype_game.columns)

forCurrCom_EN = pd.merge(std_id_median_played_n_duration, currComp_df_f_EN_llevel_cat_actype_game, how="inner", left_on=["student_id"], right_on = ["student_id"]) 
forCurrCom_EN.columns

sns.pairplot(forCurrCom_EN[["student_year_group_x", "student_gender_x", "current_competence","metadata_navigo_games_played","metadata_navigo_games_duration (minutes)","duration_per_game (minutes)"]], 
                 hue="student_year_group_x", palette="Set2", diag_kind="kde", height=5.0)#, hue_order = ["male","female"])

plt.figure(figsize=(15,10))
correlations = forCurrCom_EN.corr(method = "pearson")
correlations
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);

###############################################################################
# For sequences of games (ABA), connect game logs via [activity id = game_activity_id]
my_precious = std_f_c_act_clean_game_EN
#my_precious.to_csv('my_precious.csv',index=False)

game_logs_selected = game_logs[["game_activity_id","start_time"]]
game_logs_selected.columns

std_f_c_act_clean_game_EN.columns
std_f_c_act_clean_game_EN_selected = std_f_c_act_clean_game_EN[["student_id","activity_id","activity_type_code"]]

EN_for_seq = pd.merge(std_f_c_act_clean_game_EN_selected, game_logs_selected, how="inner", left_on=["activity_id"], right_on = ["game_activity_id"]) 
EN_for_seq_id_act = EN_for_seq[["student_id","activity_type_code","start_time"]]
EN_for_seq_id_act.to_csv('EN_for_seq.csv',index=False)



















