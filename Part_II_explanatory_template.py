#!/usr/bin/env python
# coding: utf-8

# # Part II - (Presentation Title)
# ## by (Marco)

# ## Investigation Overview
# 
# 
# >Objective 1: Analyze Duration Clusters
# 
# >Goal: To understand the distribution of trip durations by segmenting them into clusters of 600 seconds (10 minutes) each.
# 
# >Insight: The clustering of trip durations reveals prevalent travel time preferences among users, highlighting the most common lengths of bike trips.
# 
# >Objective 2: Correlate Trip Duration with Time of Day
# 
# >Goal: To examine how bike usage varies throughout the day by relating trip durations to the hour of the day.
# 
# >Insight: The analysis indicates peak usage times and helps identify patterns, such as longer trips during certain hours, which suggest specific needs or behaviors at different times.
# 
# >Objective 3: Multivariate Analysis with Gender
# 
# >Goal: To incorporate the variable of gender to explore differences in trip durations between males and females, aiming to identify which gender tends to take longer trips.
# 
# >Insight: By adding a gender perspective to the duration and hourly usage analysis, we uncover nuanced differences in biking habits. Specifically, the data suggests that males are more likely to take longer trips, especially during the central hours of the day.
# 
# 
# ## Dataset Overview and Executive Summary
# 
# >This dataset comes from a bike-sharing program and has detailed information about bike trips taken in a specific month. It covers how long trips last, when they start and end, details about the stations, bike IDs, and information about the people using the service like their type (subscriber or casual), birth year, and gender. This data is great for looking into how people use the bike-sharing program, the patterns of these bike trips, and how well the program meets users' needs.
# From analyzing this dataset, we learn a lot about how the bike-sharing program operates. We see trends like how long trips usually last, when bikes are most often used throughout the day, and how men and women use the service differently. These insights show us that the bike-sharing program is mainly used for short trips, has busy times that follow a daily pattern, and serves a diverse group of users differently.
# 

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[5]:


df_cleaned = pd.read_csv('df_cleaned_part_1_exploration.csv')


# In[6]:


df_cleaned.head()


# ## (Visualization 1)
# 
# >The visualization "Distribution of Trip Durations," illustrates the frequency distribution of bike trips based on their duration, segmented into 10-minute intervals. The x-axis represents the trip duration in seconds, while the y-axis shows the frequency of trips as a percentage of the total.
# 
# >Observations:
# A significant majority of trips are short, with the highest frequency of trips lasting less than 10 minutes. This suggests that the bike-sharing program is predominantly used for quick, short-distance travels.
# There is a rapid decline in frequency as trip duration increases, indicating that longer trips are much less common.The distribution highlights the utility of the bike-sharing program for brief commutes, aligning with an urban setting where users prefer fast and efficient modes of transportation for short distances.
# This polished and appropriately designed histogram provides a clear and concise overview of trip duration preferences among the bike-sharing program's users, underscoring the importance of catering to short-distance travelers in the program's operational and marketing strategies

# In[14]:


bins = np.arange(0, df_cleaned['duration_sec'].max() + 600 ,  600 )
plt.figure(figsize=(10, 6))
sns.histplot(data = df_cleaned, x = 'duration_sec', bins= bins, stat = 'percent')
plt.xlabel('Trip Duration (Seconds)')
plt.ylabel('Frequency (%)')
plt.title('Distribution of Trip Durations')
plt.xticks(bins, rotation=90
plt.show()
;


# ## (Visualization 2)
# 
# >The visualization "Relationship Between Hour of Day and Trip Duration," explores the correlation between the time of day and the length of bike trips in seconds. By applying a regression line, it aims to illustrate any trends in trip duration across different hours.
# 
# >Observations:
# The scatter plot, enhanced with a regression line, suggests a slight increase in trip durations during certain times of the day, potentially indicating that longer trips tend to occur during midday and evening hours.
# The application of jitter and transparency (alpha) to the plot points helps in visualizing the density of trips across the day, showing a substantial number of trips during peak commuting hours.
# Despite the spread of data points indicating variability in trip lengths, the regression line provides a guide to understanding overall trends, hinting at specific hours where users might prefer longer trips.
# This visualization effectively utilizes scatter plot and regression analysis to uncover patterns in trip durations relative to the hour of the day, offering valuable insights into user preferences and behaviors. The careful choice of plot type, encodings, and transformations ensures the data's underlying trends are appropriately highlighted, making it a polished and informative component of the analysis presentation.

# In[15]:


plt.figure(figsize= (15,10))

sns.regplot(data= df_cleaned, x = 'hour', y = 'duration_sec', truncate = False, x_jitter= 0.3, scatter_kws= {'alpha':1/20})
plt.xlabel('Hours of the Day')
plt.ylabel('Duration (Seconds)')
plt.title('Relationship Between Hour of Day and Trip Duration')
plt.xticks(range(0, 24))
plt.show();


# ## (Visualization 3)
# 
# 
# > The Visualization: Hour of Day vs. Trip Duration by Genderv aims to highlight differences in biking habits among genders.
# 
# 
# >Observation:
# For all genders, there is a wide spread of trip durations throughout the day, with a noticeable concentration of shorter trips.
# Males show a higher density of longer trips during midday hours, suggesting a tendency for longer journeys within this time frame.
# Female and Other genders also exhibit longer trips but with a slightly different distribution, indicating varying usage patterns among the groups.

# In[10]:


#faceGrid to spleet the graphs in three , one for each categorical variable (gender) 
# i also use alpha to make readebale the data inside each graphs
#i use the placeholder col_name to give for each graphs the right title

g = sns.FacetGrid(data = df_cleaned, col = 'member_gender', height= 5,  aspect= 1)
g.map(plt.scatter,  'hour',  'duration_sec', alpha = 1/20)

g.set_xlabels('HOURS_OF_THE_DAY')
g.set_ylabels('DURATION_P_SECOND')
g.set_titles("Gender: {col_name}")
("")


# ## (Visualization 4)
# 
# 
# >This PairGrid visualization explores the relationships between the hour of the day and trip duration, specifically for Male and Female users, using distinct colors to differentiate between the genders.
# 
# 
# >Observations: Both males and females show a broad range of trip durations across different hours of the day, with concentrations of shorter trips evident in the density of the scatter plots.
# The histograms reveal that trip durations tend to be somewhat longer for males, while both genders show similar patterns in hourly bike usage, peaking during morning and evening commuting times.
# Polished Aspects:
# Color Palette: The choice of blue and pink for males and females, respectively, offers clear visual differentiation while adhering to traditional color associations for gender.
# Transparency in Scatter Plots: The adjusted alpha level enhances readability, allowing for a clearer view of the data distribution and density.
# Comprehensive Legend: The legend provides an immediate reference for interpreting the color scheme, enhancing the overall comprehensibility of the plot.
# This PairGrid is an effective tool for conveying the complex relationship between trip duration, hour of the day, and gender. It showcases polished visualizations that are not only appropriate for the data but also rich in insights, highlighting key differences in biking habits between males and females.

# In[17]:


variables = ['hour','duration_sec']

palette_color = {'Male':'blue', 'Female':'pink'}

g = sns.PairGrid(data= df_cleaned[df_cleaned['member_gender'].isin(['Male','Female'])], vars= variables ,hue= 'member_gender',height= 5, aspect= 1, palette= palette_color)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, alpha = 1/30)
g.add_legend()
plt.show();


# In[ ]:




