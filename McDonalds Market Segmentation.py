#!/usr/bin/env python
# coding: utf-8

# In[29]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#import dataset
df = pd.read_csv("mcdonalds.csv")


# In[4]:


#Display variable name
df.columns


# In[5]:


#Display sample size
df.shape


# In[6]:


#Display first 3 rows of data
df.head(3)


# First we extract the first eleven columns from the data set because these columns
# contain the segmentation variables, and convert the data to a matrix. Then we
# identify all YES entries in the matrix. This results in a logical matrix with entries
# TRUE and FALSE. Adding 0 to the logical matrix converts TRUE to 1, and FALSE
# to 0. We check that we transformed the data correctly by inspecting the average
# value of each transformed segementation variable
# 

# In[7]:


MD_x = df.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
col_means = np.round(np.mean(MD_x, axis = 0), 2)
print (col_means)


# In[8]:


#Label encoding for categorical - Converting 11 cols with yes/no

from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy','convenient','spicy','fattening','greasy','fast','cheap','tasty',
              'expensive','healthy','disgusting']

for i in cat:
    labelling(i)
df


# In[9]:


#Considering only first 11 attributes
df_eleven = df.loc[:,cat]
df_eleven


# In[10]:


#Considering only the 11 cols and converting it into array
x = df.loc[:,cat].values
x


# In[11]:


#Principal Componen Analysis

from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components = 11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)
pf


# In[12]:


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


# In[13]:


np.cumsum(pca.explained_variance_ratio_)


# In[14]:


pca = PCA()
pca.fit(df_eleven)

#Get loadings and number of peincipal components
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC" + str(i) for i in range(1, num_pc+1)]
loadings_df = pd.DataFrame(loadings.T, columns = pc_list)
loadings_df['variable'] = df_eleven.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[18]:


#Principal Components Analysis of the fast food data set

get_ipython().system('pip install bioinfokit')
from bioinfokit.visuz import cluster

#get PC scores
pca_scores = PCA().fit_transform(x)

#get 2D biplot
cluster.biplot(cscore = pca_scores, loadings = loadings, labels = df.columns.values, var1 = round (pca.explained_variance_ratio_[0]*100, 2), var2 = round(pca.explained_variance_ratio_[1]*100, 2), show = True, dim = (10,5))


# # EXTRACTING SEGMENTS

# In[21]:


pip install yellowbrick


# In[22]:


#Extracting Segments

#Using k-means clustering analysis
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer (model, k = (1,12)).fit(df_eleven)
visualizer.show()


# In[24]:


#K-means clustering

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0).fit(df_eleven)
df['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares
print (kmeans.n_iter_) #number of iterations that k-means algorithm rns to get a minimum within-cluster sum of squares
print (kmeans.cluster_centers_) #Location of the centroids on each cluster.


# In[25]:


#To see each cluster size
from collections import Counter
Counter(kmeans.labels_)


# In[30]:


#Visualizing clusters
sns.scatterplot(data = pf, x = 'pc1', y = 'pc2', hue = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker = 'X', c = 'r', s = 80, label = 'centroids')
plt.legend()
plt.show()


# # DESCRIBING SEGMENTS

# In[36]:


#DESCRIBING SEGMENTS

from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(df['cluster_num'],df['Like'])
#Reordering cols
crosstab = crosstab[['I hate it!-5','-4','-3','-2','-1','0','+1','+2','+3','+4','I love it!+5']]
crosstab 


# In[37]:


#Mosaic Plot

plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab.stack())
plt.show()


# In[38]:


#Mosaic plot gender vs segment
crosstab_gender = pd.crosstab(df['cluster_num'], df['Gender'])
crosstab_gender


# In[39]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[41]:


#box plot for age

sns.boxplot(x='cluster_num', y='Age', data = df)


# # SELECTING TARGET SEGMENT

# In[42]:


#Calculating the mean
#Visit Frequency 
df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit = df.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[44]:


#Like
df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[46]:


#Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
Gender = df.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[47]:


segment = Gender.merge(Like, on = 'cluster_num', how = 'left').merge (visit, on = 'cluster_num', how='left')
segment


# In[49]:


#Target Segments

plt.figure(figsize = (9,4))
sns.scatterplot(x = 'VisitFrequency', y = 'Like', data = segment, s=400, color='r')
plt.title("Simple segment evaluation plot for the fast food data set", fontsize = 15)
plt.xlabel('Visit', fontsize = 12)
plt.ylabel('Like', fontsize = 12)
plt.show()


# In[ ]:




