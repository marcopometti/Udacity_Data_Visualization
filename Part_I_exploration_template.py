#!/usr/bin/env python
# coding: utf-8

# # Part I - (Dataset Exploration Title)
# ## by (MARCO)
# 
# ## Introduction

# ## Preliminary Wrangling

# In[2]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# In[3]:


df = pd.read_csv('201902-fordgobike-tripdata.csv')


# In[5]:


# get info from my database
df.info()

#identify null values in my dataset
df_unique = df.isnull().sum()
df_unique


# In[6]:


#delete null_values from dataset
df_final = df.dropna()


# In[7]:


#get some info of my features using describe methonds

df_final.describe()


# ### What is the structure of your dataset?
# 
# > Number of rows , 183.412, Number of colomuns 16 columns, dtypes: float64(7), int64(2), object(7)
# 
# 
# > 8.265 null values on member_gender and member_birth_year , i dediced to delete them for the purpose of the        analysis 
#   
#   
# >  We have all the info reletated to gender and data of birth of the persone who rent the bike, start and end station , duration of the trip , the id of the bike , latitude and longitude of the station and the usere type "Customer" o "Subscriber".
#   
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# > User Type (user_type): By distinguishing between "Customer" and "Subscriber", you can explore how different groups use the service. This may include usage frequency, average trip duration, and preferences for timing or station, providing insights for marketing strategies and service development.
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > Duration of the trip, User type, bike_id, gender and birth_date

# ## Univariate Exploration

# In[8]:


#create a subset of my dataframe taking only the columns i needed and 
#i create one more columns to have the info of the age per member and also the day of the trip

df_final = df_final[['duration_sec','start_time','end_time','bike_id','user_type','member_birth_year','member_gender','bike_share_for_all_trip']]

df_final['AGE'] = 2019 - df_final['member_birth_year']
df_final['start_time'] = pd.to_datetime(df_final['start_time'])
df_final['hour'] = df_final['start_time'].dt.hour

df_final = df_final[df_final['AGE'] <= 80]


# In[9]:


#create a counterplot using seaborn package to understand how may rides are done by  subscribers or customers 

sns.countplot(data = df_final, x = 'user_type')
plt.xlabel('USER_TYPE')
plt.ylabel('RIDE_PER_USER_TYPE')
plt.title('RIDE_PER_USER_TYPE')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'.replace(',', '.')))
("")


# In[10]:


#clean the dabase and take just the rides under 1h of trip. 

df_cleaned = df_final[df_final['duration_sec']<=3600]
df_cleaned


# In[11]:


#create the bins for the distribution. I decided to split the duration into bins of 10 minutes 
bins = np.arange(0, df_cleaned['duration_sec'].max() + 600 ,  600 )


# In[28]:


#create a histogram using seaborn package to represent the distribution of the Bike_Trip
plt.figure(figsize=(10, 6))
sns.histplot(data = df_cleaned, x = 'duration_sec', bins= bins, stat = 'percent')
plt.xlabel('TRIP_DURATION')
plt.ylabel('FREQUENCY_%')
plt.title('TRIP_DISTRIBUTION')
("")


# > We can see some outlier points in duration of the trip. 1.3K rows out of 173K have a duration trips greater than 1H.. I had to remove this outlier to show the real distribution and plot the number in the correct way. 
# > i removed null values , take only the features needed for my analysis and only for the trip distribution , i had to subset the dataframe removing all the records greater than 1h of trip duration

# ## Bivariate Exploration

# In[51]:


#REGRESSION PLOT , TO UNDERSTAND IF THERE IS A CORRELATION BETWEEN HOURS AND DURATION OF THE TRIP.

plt.figure(figsize= (15,10))

sns.regplot(data= df_cleaned, x = 'hour', y = 'duration_sec', truncate = False, x_jitter= 0.3, scatter_kws= {'alpha':1/20})
plt.xlabel('HOURS_OF_THE_DAY')
plt.ylabel('DURATION_P_SECOND')
plt.title('REGRESSION_PLOT');


# In[13]:


#SCATTER PLOT TO SEE IF THERE ARE SOME CORRELATION BETWEEN HOURS AND DURATION OF THE TRIP
plt.figure(figsize= (10,6))
sns.scatterplot(data = df_cleaned, x = 'hour', y = 'duration_sec');
plt.xlabel('HOURS_OF_THE_DAY')
plt.ylabel('DURATION_P_SECOND')
plt.title('REGRESSION_PLOT');


# In[14]:


#BOXPLOT TO UNDERSTAND BETTER THE CORRELATION OF OUR VARIABLES

plt.figure(figsize= (10,6))
sns.boxplot(data = df_cleaned, x = 'hour', y = 'duration_sec', color= 'tab:blue')
plt.xlabel('HOURS_OF_THE_DAY')
plt.ylabel('DURATION_P_SECOND')
plt.title('BOXPLOT');
("")


# In[15]:


#HEATMAP TO UNDERSTAND THE PEAK USAGE HOURS

plt.figure(figsize= (15,6), dpi=100)


bin_x = np.arange(0, 23 + 1 , 1)
bin_y = np.arange(0, 3600 + 600, 600)

h2d = plt.hist2d(data = df_cleaned, x = 'hour', y = 'duration_sec', cmin = 600, cmap = 'viridis_r', bins= [bin_x,bin_y])

plt.colorbar()
plt.xlabel('HOURS_OF_THE_DAY')
plt.ylabel('DURATION_P_SECOND')
plt.title('HEATMAP');

#i inserted with this code the count of values for each bins that i have generated

count = h2d[0]

for i in range(count.shape[0]):
    for j in range(count.shape[1]):
        c = count[i,j]
        if c > 6000:
            plt.text(bin_x[i] + 0.5, bin_y[j] + 300, int(c), ha = 'center', va = 'center', color = 'white')
            
        elif c > 0:
            plt.text(bin_x[i] + 0.5, bin_y[j] + 300, int(c), ha = 'center', va = 'center', color = 'black')

plt.show()
("")


# > I investigated the relationship between the time of day and trip duration for a bike-sharing service. Through various visualizations, I discovered that peak usage hours are at 8 AM and 4 PM, coinciding with common commuting times to and from the office. Additionally, regression plots revealed that during these peak hours, users tend to utilize bikes for longer periods, whereas off-peak hours are characterized by shorter usage durations.

# ## Multivariate Exploration
# 
# > Create plots of three or more variables to investigate your data even
# further. Make sure that your investigations are justified, and follow from
# your work in the previous sections.
# 
# > **Rubric Tip**: This part (Multivariate Exploration) should include at least one Facet Plot, and one Plot Matrix or Scatterplot with multiple encodings.
# 
# >**Rubric Tip**: Think carefully about how you encode variables. Choose appropriate color schemes, markers, or even how Facets are chosen. Also, do not overplot or incorrectly plot ordinal data.

# In[16]:


#faceGrid to spleet the graphs in three , one for each categorical variable (gender) 
# i also use alpha to make readebale the data inside each graphs
#i use the placeholder col_name to give for each graphs the right title

g = sns.FacetGrid(data = df_cleaned, col = 'member_gender', height= 5,  aspect= 1)
g.map(plt.scatter,  'hour',  'duration_sec', alpha = 1/20)

g.set_xlabels('HOURS_OF_THE_DAY')
g.set_ylabels('DURATION_P_SECOND')
g.set_titles("Gender: {col_name}")
("")


# In[25]:


#create variables list to pass it as a parameter in PairGrid object
#use a subset of my dataframe to take just two categorical variable (gender male and female) because i decided to use hue parameter to color different points based on the categorical variable


variables = ['hour','duration_sec']

palette_color = {'Male':'blue', 'Female':'pink'}

g = sns.PairGrid(data= df_cleaned[df_cleaned['member_gender'].isin(['Male','Female'])], vars= variables ,hue= 'member_gender',height= 5, aspect= 1, palette= palette_color)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, alpha = 1/30)
g.add_legend()
plt.show()
("")


# > While exploring the interactions between gender, hours of the day, and trip duration, an interesting pattern    emerged that challenges conventional expectations. Notably, the finding that males tend to undertake longer trips primarily in the middle part of the day suggests a deviation from the typical rush-hour commuting pattern often associated with work-related travel. This could imply a diverse range of trip purposes among males, including midday errands or leisure activities.
# On the other hand, the observation that females engage in shorter trips, with a higher concentration in the early and late hours, suggests a potential alignment with traditional commuting times. However, the shorter trip duration could also indicate a preference or necessity for closer destinations or perhaps a more efficient use of the service for specific types of errands or tasks that are time-bound to the beginning or end of the day.
# These patterns reveal intriguing gender-based differences in service usage and suggest the influence of various social, economic, and possibly familial roles on travel behavior. The divergence from expected norms, particularly the non-peak usage times for longer trips among males, opens up avenues for further investigation into the underlying reasons for these behaviors and how they might inform improvements to transportation planning and policy.

# ## Conclusions
# >Throughout the data exploration process focusing on gender, hours of the day, and trip duration, we've uncovered several key insights that shed light on the usage patterns of a transportation service. Here's a summary of the main findings:
# 
# >Gender-Based Trip Duration Differences: There's a noticeable difference in trip durations between genders, with males generally taking longer trips, especially in the middle of the day.
# 
# >Temporal Concentration of Female Trips: Female users tend to make shorter trips, but these trips are more densely concentrated in the early and late hours of the day.
# 
# >Non-Peak Hour Preference for Longer Trips by Males: An unexpected pattern emerged showing males prefer taking longer trips during midday, outside of the typical morning and evening rush hours. This suggests a variety of trip purposes beyond commuting to work.
# 
# >Potential Efficiency in Female Trip Planning: The shorter duration and specific timing of trips made by females could imply a strategic approach to trip planning, possibly to accommodate other daily responsibilities or preferences.
# 
# >Reflecting on the Exploration Steps:
# Initial Data Examination: The exploration began with an initial examination of the dataset, identifying relevant variables (gender, hours, and duration) for analysis. This step was crucial for understanding the structure of the data and formulating the direction of the analysis.
# 
# >Visualization: Using visual tools like scatter plots and histograms within a PairGrid enabled the observation of relationships and patterns across multiple dimensions. This approach was instrumental in identifying the nuanced differences in trip behaviors between genders and across different times of the day.
# 
# >Statistical Summaries: While not explicitly mentioned, incorporating statistical summaries could further support the visual findings, offering a quantitative backing to the observed patterns.
# 
# >Consideration of External Factors: The exploration considered how external factors (e.g., social roles, employment patterns) might influence the observed patterns. This reflection is crucial for understanding the context behind the data and for framing further research.
# 
# >Conclusion and Further Research:
# The exploration revealed interesting patterns in transportation service usage, highlighting gender differences in trip timing and duration. These insights not only contribute to a better understanding of user behavior but also suggest areas for further research. For instance, investigating the reasons behind the midday preference for longer trips among males or the efficiency in trip planning among females could provide valuable information for transportation service providers and urban planners. Additionally, exploring other demographic variables or external factors like weather conditions and urban infrastructure could offer a more comprehensive view of the dynamics influencing transportation service usage.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




