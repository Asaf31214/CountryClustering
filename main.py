#this is the file the project is first written on. the /notebooks/main.ipynb file is the
#cleaner version of same code.

import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore
import seaborn as sns
#matplotlib.use("webagg")
dataset= pd.read_csv("data/Kaggle.csv")
new_datas=dataset[[
    'Id','Internet users percentage of population 2014','Carbon dioxide emissions per capita 2011 Tones','Domestic food price level 2009 2014 index','Electrification rate or population','Gender Inequality Index 2014','Gross domestic product GDP percapta','Homicide rate per 100k people 2008-2012','Mean years of schooling - Years','Prison population per 100k people','Human Development Index HDI-2014']]
final_set=new_datas.rename(columns={'Id':'Country_name','Internet users percentage of population 2014':'Internet_user_percentage','Carbon dioxide emissions per capita 2011 Tones':'C02_emission_per_capita','Domestic food price level 2009 2014 index':'Food_price_level','Electrification rate or population':'Electrification_rate','Gender Inequality Index 2014':'Gender_inequality_index','Gross domestic product GDP percapta':'GDP_per_capita','Homicide rate per 100k people 2008-2012':'Homicide_rate','Mean years of schooling - Years':'Mean_schooling_years','Prison population per 100k people':'Prisoner_rate','Human Development Index HDI-2014':'Human_development_index'})

final_numeric=final_set[['Internet_user_percentage', 'C02_emission_per_capita',
       'Food_price_level', 'Electrification_rate', 'Gender_inequality_index',
       'GDP_per_capita', 'Homicide_rate', 'Mean_schooling_years',
       'Prisoner_rate', 'Human_development_index']]
normalized_set=final_numeric.apply(zscore)
# z score normalization
#normalized_set=(final_numeric-final_numeric.min())/(final_numeric.max()-final_numeric.min())
# 0-1 normalization
#normalized_set.insert(0,"Country_name",final_set.Country_name,True)
# wcss=[]
# for i in range(1,21):
#     kmeans=KMeans(n_clusters=i,random_state=42)
#     kmeans.fit(normalized_set)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 21), wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# 6 clusters are optimum


# sns.heatmap(normalized_set.corr(), annot=True,cmap="YlGnBu")
# plt.show()
# correlation check, nothing sus

kmeans=KMeans(n_clusters=6,random_state=0)
kmeans.fit(normalized_set)
print(kmeans.labels_)
pca=PCA(n_components=2,whiten=True)
pca.fit(normalized_set)
normalized_set_pca=pca.fit_transform(normalized_set)
print("variance ratio: ", pca.explained_variance_ratio_)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("number of components")
# plt.ylabel("cumulative explained varience")
# plt.show()

nspca=pd.DataFrame(data=normalized_set_pca, columns=("x_loc","y_loc"))
labels=pd.Series(data=kmeans.labels_)

plt.scatter(x=nspca["x_loc"],y=nspca["y_loc"],c=labels)
plt.show()

# principalDF=pd.DataFrame(data = normalized_set_pca, columns = ['principal component 1', 'principal component 2'])
# plt.scatter(principalDF["principal component 1"],principalDF["principal component 2"])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

