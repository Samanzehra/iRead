# first get and set the working directory
from os import chdir, getcwd
getcwd()
chdir('C:\\***\\iread_analysis')
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

###############################################################################
#read logs
logs = pd.read_csv('C:\\***\\analytics.logs.csv')
print(logs.columns)
# read student_users
student_users = pd.read_csv('C:\\***\\analytics.student_users.csv')
print(student_users.columns)
# read questions
questions = pd.read_csv('C:\\***\\analytics.questions.csv')
print(questions.columns)
#read game_logs
game_logs = pd.read_csv('C:\\***\\analytics.game_logs.csv')
print(game_logs.columns)
#read events
events = pd.read_csv('C:\\***\\analytics.events.csv')
print(events.columns)
#read game_event_type
game_event_type = pd.read_csv('C:\\***\\analytics.game_event_type.csv')
print(game_event_type.columns)
#read question_content
question_content = pd.read_csv('C:\\***\\analytics.question_content.csv')
print(question_content.columns)
#read EN_student_id_only_379 
EN_student_id_only_379 = pd.read_csv('C:\\***\\EN_student_id_only_379.csv')
#read games
games = pd.read_csv('C:\\***\\analytics.games.csv')
print(games.columns)
#read activities 
activities = pd.read_csv('C:\\***\\analytics.activities.csv')
print(activities.columns)
#read features
features = pd.read_csv('C:\\***\\analytics.features.csv')
print(features.columns)
#read models
models = pd.read_csv('C:\\***\\analytics.models.csv')
print(models.columns)
#read activity types
activity_types = pd.read_csv('C:\\***\\analytics.activity_types.csv')
print(activity_types.columns)
content = pd.read_csv('C:\\***\\analytics.content.csv')
print(content.columns)
df_EN_student_id_Comp = pd.read_csv('C:\\***\\df_EN_student_id_Comp.csv')

# use a function to inner join two dataframes
def combine(df1,df2,key1,key2):
    return df1.merge(df2, how = 'inner', left_on = key1, right_on = key2)
###############################################################################
feature_game_summary_EN = pd.read_csv('C:\\***\\feature_game_summary_EN.csv')
#subgroup students based on gender plus their median competence change in the games
df_EN_student_id_Comp.columns
df_EN_student_id_Comp_median = df_EN_student_id_Comp.groupby(by = ['student_id', 'student_year_group','student_gender'], as_index = False)['competence_change'].median().rename(columns={'competence_change':'median_competence_change'})
df_EN_student_id_Comp_median.columns
###############################################################################
#df_EN_female_high = df_EN_student_id_Comp_median[(df_EN_student_id_Comp_median['student_gender'] == 'female') & (df_EN_student_id_Comp_median['median_competence_change'] >= 3)]
#df_EN_female_low = df_EN_student_id_Comp_median[(df_EN_student_id_Comp_median['student_gender'] == 'female') & (df_EN_student_id_Comp_median['median_competence_change'] < 3)]
#df_EN_male_high = df_EN_student_id_Comp_median[(df_EN_student_id_Comp_median['student_gender'] == 'male') & (df_EN_student_id_Comp_median['median_competence_change'] >= 3)]
#df_EN_male_low = df_EN_student_id_Comp_median[(df_EN_student_id_Comp_median['student_gender'] == 'male') & (df_EN_student_id_Comp_median['median_competence_change'] < 3)]

###############################################################################
#Connect questions with game log to log to student id  in df_EN_student_id_Comp_median
question_game_logs = combine(questions, game_logs,'question_game_log_id','game_log_id')
question_game_logs.columns
#keep only needed columns
question_game_logs = question_game_logs[['question_id', 'question', 'question_number','game_log_id', 'start_time','game_activity_id']]
# combine with logs so that combine with student_id
question_game_logs_logs = combine(question_game_logs, logs, 'game_log_id', 'log_id') 
question_game_logs_logs.columns
#keeping only needed column
question_game_logs_logs = question_game_logs_logs[['question_id', 'question', 'question_number', 'game_log_id',
       'start_time', 'game_activity_id', 'log_id', 'log_user_id']]

question_game_logs_logs_student_id = combine(question_game_logs_logs, df_EN_student_id_Comp_median, 'log_user_id','student_id')
question_game_logs_logs_student_id.columns

question_game_logs_logs_student_id_events = combine(question_game_logs_logs_student_id, events, 'question_id','event_question_id')
question_game_logs_logs_student_id_events.columns

# cobine game_event_type
question_game_logs_logs_student_id_events_game_event_type

# combine this with feature_game_summary_EN
question_game_logs_logs_features = combine(question_game_logs_logs, feature_game_summary_EN, 'game_activity_id','activity_id')
question_game_logs_logs_features.columns
#selected columns
question_game_logs_logs_features = question_game_logs_logs_features[['question_id', 'question', 'question_number', 'game_log_id',
       'start_time', 'game_activity_id', 'log_user_id', 'feature_model_id','linguistic_level', 'category','feature_difficulty', 'model_id',
       'model_name', 'activity_id', 'activity_iread_id', 'input_type',
       'activity_difficulty', 'activity_feature_id', 'activity_type',
       'activity_game_id', 'activity_type_id', 'activity_type_name',
       'activity_type_code', 'game_id', 'game_name'
        ]]

# combine with qc content and then events 
question_game_logs_logs_features_qc_content = combine(question_game_logs_logs_features,question_content,'question_id','qc_question_id')
# now combine with event via event_content_id
question_game_logs_logs_features_qc_content_events = combine(question_game_logs_logs_features_qc_content,events,'qc_content_id','event_content_id')
# combine with game_event_type
question_game_logs_logs_features_qc_content_events_game_event_type = combine(question_game_logs_logs_features_qc_content_events,game_event_type,'event_type','game_event_id')
question_game_logs_logs_features_qc_content_events_game_event_type.columns

df_rough1 = question_game_logs_logs_features_qc_content_events_game_event_type[['question_id', 'question', 'question_number', 'game_log_id',
       'log_user_id', 'linguistic_level', 'category', 'feature_difficulty',
       'activity_difficulty', 'activity_type',
       'activity_game_id', 'activity_type_id', 'activity_type_name',
       'activity_type_code', 'game_id', 'game_name', 'qc_question_id',
       'qc_content_id', 'qc_content_multiplicity', 'event_id',
       'event_timestamp', 'event_source', 'event_data', 'event_comment',
       'event_question_id', 'event_type', 'event_content_id', 'game_event_id',
       'game_event_type']]

# now combine question_game_logs_logs_features with df_EN_student_id_Comp_median on student_id

df_rough2 = combine(df_rough1,df_EN_student_id_Comp_median,'log_user_id','student_id')
df_rough2.columns
df_rough = df_rough2[['event_id','student_id', 'question_id', 'question_number', 'game_log_id',
       'linguistic_level', 'category', 'game_name', 'event_type', 'student_year_group', 'student_gender',
       'median_competence_change']]

df1_forBKT = df_rough.replace({'event_type' :4},0)
df1_forBKT = df1_forBKT.replace({'event_type' :3},1)
df1_forBKT.columns
df1_forBKT.head()
#df1_forBKT.to_csv('C:\\***\\df1_forBKT_updated.csv', index=False)

#log_for_EN_students = combine(logs, EN_student_id_only_379, 'log_user_id', 'student_id')
#log_for_EN_students = log_for_EN_students[['log_id','log_user_id']]
#game_log_for_EN_students = combine(log_for_EN_students, game_logs, 'log_id', 'game_log_id')
#game_log_for_EN_students.columns
#game_log_for_EN_students = game_log_for_EN_students[['game_log_id', 'log_user_id', 'start_time','game_activity_id']]
#game_log_for_EN_students_activities = combine(game_log_for_EN_students,activities, 'game_activity_id','activity_id')
#game_log_for_EN_students_activities.columns
#game_log_for_EN_students_activities =  game_log_for_EN_students_activities[['game_log_id','log_user_id','start_time', 'game_activity_id',
#       'activity_id', 'activity_feature_id', 'activity_type', 'activity_game_id' ]]
#game_log_for_EN_students_activities_games = combine(game_log_for_EN_students_activities, games, 'activity_game_id','game_id')
#game_log_for_EN_students_activities_games.columns
#game_log_for_EN_students_activities_games[['game_log_id', 'log_user_id', 'start_time', 'game_activity_id','activity_id', 'activity_feature_id', 'activity_type','game_name']]
#game_log_for_EN_students_activities_games_features = combine(game_log_for_EN_students_activities_games, features, 'activity_feature_id','feature_id')
#game_log_for_EN_students_activities_games_features.columns

#game_log_for_EN_students_activities_games_features = game_log_for_EN_students_activities_games_features[['game_log_id', 'log_user_id', 'start_time', 'game_activity_id',
#       'activity_id', 'activity_feature_id', 'activity_type',
#       'activity_game_id', 'game_id', 'game_name', 'feature_id','linguistic_level', 'category']]

#game_log_for_EN_students_activities_games_features_content = combine(game_log_for_EN_students_activities_games_features, content, 'feature_id','content_feature_id')
#game_log_for_EN_students_activities_games_features_content.columns
#game_log_for_EN_students_activities_games_features_content = game_log_for_EN_students_activities_games_features_content[['game_log_id', 'log_user_id', 'start_time', 
#      'activity_type','game_name','linguistic_level', 'category', 'content_id', 'content_feature_id']]
#game_log_for_EN_students_activities_games_features_content_events = combine(game_log_for_EN_students_activities_games_features_content, events, 'content_id','event_content_id')
#game_log_for_EN_students_activities_games_features_content_events.columns
#game_log_for_EN_students_activities_games_features_content_events = game_log_for_EN_students_activities_games_features_content_events[['game_log_id', 'log_user_id', 'start_time', 'activity_type',
#       'game_name', 'linguistic_level', 'category', 'event_id', 'event_question_id', 'event_type']]
#
#game_log_for_EN_students_activities_games_features_content_events_questions = combine(game_log_for_EN_students_activities_games_features_content_events, questions, 'game_log_id','question_game_log_id')
#game_log_for_EN_students_activities_games_features_content_events_questions.columns
#game_log_for_EN_students_activities_games_features_content_events_questions = game_log_for_EN_students_activities_games_features_content_events_questions[['game_log_id', 'log_user_id',
#       'game_name', 'linguistic_level', 'category', 'event_id', 'event_type', 'question_id']]

#df_1 = game_log_for_EN_students_activities_games_features_content_events_questions[['event_id',
#                                                                                    'log_user_id',
#                                                                                    'category',
#                                                                                    'game_name',
#                                                                                    'event_type']]

### read the prepeared data
df1_forBKT = pd.read_csv('C:\\***\\df1_forBKT_selectedGames.csv')
df1_female = df1_forBKT[(df1_forBKT['student_gender'] == 'female')]
df1_female.columns
df1_male = df1_forBKT[(df1_forBKT['student_gender'] == 'male')]
df1_male.columns
df1_male.nunique()
#subsets
df1_female_high = df1_forBKT[(df1_forBKT['student_gender'] == 'female') & (df1_forBKT['median_competence_change'] >= 3)]
df1_female_high.columns
df1_female_low = df1_forBKT[(df1_forBKT['student_gender'] == 'female') & (df1_forBKT['median_competence_change'] < 3)]
df1_female_low.columns
df1_female_low = df1_forBKT[(df1_forBKT['student_gender'] == 'female') & (df1_forBKT['median_competence_change'] < 3)]
df1_male_high = df1_forBKT[(df1_forBKT['student_gender'] == 'male') & (df1_forBKT['median_competence_change'] >= 3)]
df1_male_low = df1_forBKT[(df1_forBKT['student_gender'] == 'male') & (df1_forBKT['median_competence_change'] < 3)]

df1_female_high_GPC = df1_female_high[df1_female_high['category'] == 'GPC']
df1_female_high_GPC.columns
df1_female_high_GPC = df1_female_high_GPC[['event_id', 'student_id', 'question_id', 'category', 'game_name','event_type']]
df1_female_high_GPC.nunique()
#df_1_GPC.to_csv('C:\\***\\df_1_GPC.csv', index=False)
df1_female_low_GPC = df1_female_low[df1_female_low['category'] == 'GPC']
df1_female_low_GPC.columns
df1_female_low_GPC = df1_female_low_GPC[['event_id', 'student_id', 'question_id', 'category', 'game_name','event_type']]
df1_female_low_GPC.nunique()

#df_1_GPC_new = df_1_GPC.sort_values(by = 'game_name')
#df_1_GPC_new.head(10)
#
#df_1_GPC_useful = df_1_GPC.sample(1000).sort_values(by = ['game_name','event_id'])

###############################################################################
############ BKT plain ########################################################
###############################################################################
# Install pyBKT from pip!
!pip install pyBKT

# Import all required packages including pyBKT.models.Model!
import numpy as np
import pandas as pd
from pyBKT.models import Model
import matplotlib.pyplot as plt
# Note that the seed chosen is so we can consistently
# replicate the results and avoid as much randomness
# as possible.
model = Model(seed = 42, num_fits = 1)
defaults = {'order_id': 'event_id',#'game_log_id',#'event_id',
            'user_id' : 'student_id',#log_user_id
            'skill_name' : 'category', 
            'problem_id' : 'question_id',
#            'student_class_id' : 'student_year_group_x',
            'template_id' : 'game_name',
            'correct': 'event_type'}

skill = 'Clusters'#'Irregular GPCs'#'Frequency'#'Syllabification'#'Clusters'#'GPC'

#as event is a click...there could be multiple events for the same question,
# but BKT theoretical assumption sugget to take into consideration only the first attempt results 
#in this case first 'event type (correct/incorrect)' for each question-game pair (problem-template in BKT)
# also it didnt make any difference if we set game id or event id as order id as its reading the data in order   
check = df1_female[['event_id','game_log_id','student_id','category','question_id','game_name','event_type']].drop_duplicates(subset = ['question_id',['game_name' =='hearoglyphs'])
check.describe
check.nunique()

# data.drop_duplicates(subset = ['question_id','game_name'])
model.fit(data = check, defaults = defaults, #df1_GPC_raw_forBKT
          skills = skill,
          multigs = 'game_name', multilearn = 'game_name')

params = model.params()
params
#params.to_csv('C:\\***\\params_df1_female_IrregularGPCs.csv', index=True)

# We will get warnings for using indexing past lexsort. That's fine,
# and we will disable these warnings.

params = model.params()
#####
import warnings
warnings.simplefilter(action='ignore')

# Plot the learns, forgets, slips and guesses for each of the classes.
plt.figure(figsize = (12, 6))
plt.plot(params.loc[(skill, 'guesses')], label = 'Guesses')
plt.plot(params.loc[(skill, 'slips')], label = 'Slips')
plt.plot(params.loc[(skill, 'learns')], label = 'Learns')
#plt.plot(params.loc[(skill, 'forgets')], label = 'Forgets')
plt.xlabel('Game Name')
plt.ylabel('Rate')
plt.title('BKT Parameters per game')
plt.legend();

###############################################################################
#'order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id',
#       'original', 'correct', 'attempt_count', 'ms_first_response',
 #      'tutor_mode', 'answer_type', 'sequence_id', 'student_class_id',
  #     'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name',
   #    'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time',
    #   'template_id', 'answer_id', 'answer_text', 'first_action',
     #  'bottom_hint', 'opportunity', 'opportunity_original'
