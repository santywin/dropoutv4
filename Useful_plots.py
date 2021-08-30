#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:

"""
final = pd.read_csv("../final.csv")
final.head()
final.columns
final.describe()
final.info()
"""

# In[31]:


def lineplotAll():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    fig = plt.figure(figsize = (15, 5)) # 

    ax = fig.add_subplot()

    # plot the data
    ax.plot(X,S, color = "red", alpha = 0.5, lw = 3, label = "Sine")
    ax.plot(X,C, color = "green", alpha = 0.5, lw = 3, label = "Cosine")

    # control the limits
    # you can change the limits of the x and y labels to make your plot look nicer
    plt.xlim(X.min()*1.5, X.max()*1.5)
    plt.ylim(C.min()*1.5, C.max()*1.5)

    # change the ticks
    # ticks are just a way to 'change the values' represented on the x and y axis
    plt.xticks(
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'] # r before each string means raw string
    )

    plt.yticks(
        [-1, 0, +1],
        [r'$-1$', r'$0$', r'$+1$']
    )

    # removes the right and top spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # changes the position of the other spines
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['bottom'].set_color('black') # this helps change the color
    ax.spines['bottom'].set_alpha(.3) # and adds some transparency to the spines
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_alpha(.3)

    # annotate different values

    t = 2*np.pi/3

    plt.plot([t, t], [0, np.sin(t)], color ='red', linewidth=1.5, linestyle="--", alpha = 0.5)
    plt.scatter(t, np.sin(t), 50, color ='red', alpha = 0.5)
    plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$' + "\nNotice the the arrows are optional",
                 xy=(t, np.sin(t)), 
                 xycoords='data',
                 fontsize=16,)


    plt.plot([t, t], [0, np.cos(t)], color ='green', linewidth=1.5, linestyle="--", alpha = 0.5)
    plt.scatter(t, np.cos(t), 50, color ='green', alpha = 0.5)
    plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                 xy=(t, np.cos(t)), 
                 xycoords='data',
                 xytext=(t/2, -1), 
    #              xytext=(-90, -50), 
    #              textcoords='offset points', 
                 fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # adjust the x and y ticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    # prettify
    plt.title("Sine and Cosine functions (Original by N. Rougier)")
    plt.legend(loc = "upper left");
    
#lineplotAll()


# In[51]:


def lineplotSimple(df, xlabel, ylabel, title):
    fig = plt.figure(figsize = (15, 5)) # 

    ax = fig.add_subplot()

    # plot the data
    ax.plot(df[xlabel],df[ylabel], color = "red", alpha = 0.5, lw = 3, label = ylabel)
    #ax.plot(X,C, color = "green", alpha = 0.5, lw = 3, label = "Cosine")

    # control the limits
    # you can change the limits of the x and y labels to make your plot look nicer
    #plt.xlim(df[xlabel].min()*1.5, df[xlabel].max()*1.5)
    #plt.ylim(C.min()*1.5, C.max()*1.5)

    # change the ticks
    # ticks are just a way to 'change the values' represented on the x and y axis
    """
    plt.xticks(
        [df[xlabel], df[xlabel]],
        [r'$min(df[xlabel])$', r'$max(df[xlabel])$'] # r before each string means raw string
    )

    plt.yticks(
        [min(df[ylabel]), max(df[ylabel])],
        [r'$min(df[ylabel])$', r'$max(df[ylabel])$']
    )
"""
    
    # removes the right and top spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # changes the position of the other spines
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['bottom'].set_color('black') # this helps change the color
    ax.spines['bottom'].set_alpha(.3) # and adds some transparency to the spines
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_alpha(.3)

    # annotate different values

    t = 2*min(df[ylabel])/3

    plt.plot([t, t], [0, np.sin(t)], color ='red', linewidth=1.5, linestyle="--", alpha = 0.5)
    """plt.scatter(t, np.sin(t), 50, color ='red', alpha = 0.5)
    #plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
                 xy=(t, np.sin(t)), 
                 xycoords='data',
                 fontsize=16,)

    """
    plt.plot([t, t], [0, np.cos(t)], color ='green', linewidth=1.5, linestyle="--", alpha = 0.5)
    plt.scatter(t, np.cos(t), 50, color ='green', alpha = 0.5)
    """plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                 xy=(t, np.cos(t)), 
                 xycoords='data',
                 xytext=(t/2, -1), 
    #              xytext=(-90, -50), 
    #              textcoords='offset points', 
                 fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    """
    # adjust the x and y ticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))
    plt.xticks(rotation=90)
    plt.grid()
    # prettify
    plt.title(title)
    plt.legend(loc = "upper left");
"""   
l = final.groupby('nombre_carrera', as_index = False).segMatTot.mean()
print(l.head())
lineplotSimple(l, "nombre_carrera", "segMatTot", "Número de segundas matrículas por carrera (media)")
"""

# In[35]:


def barplot(d, title):
    
    colors = [plt.cm.Spectral(i/float(len(d.keys()))) for i in range(len(d.keys()))]

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot()

    ax.bar(d.keys(), d.values(), color = colors)

    for k, v in d.items():
        ax.text(k, v*1.1, v, fontsize = 10, horizontalalignment='center', verticalalignment='bottom',rotation=90)

    ax.tick_params(axis='x', labelrotation = 90, labelsize = 12)
    ax.set_ylim(0, max(d.values()) *1.5)
    ax.set_title(title, fontsize = 14);
"""    
d = final.groupby('nombre_carrera').segMatTot.mean().to_dict()
barplot(d, "Número de segundas matrículas por carrera (media)")
"""

# In[23]:


def scatterplot(df, category, xlabel, ylabel ):
    # instanciate the figure
    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(1,1,1,)

    # ----------------------------------------------------------------------------------------------------
    # iterate over each category and plot the data. This way, every group has it's own color. Otherwise everything would be blue
    for cat in sorted(list(df[category].unique())):
        # filter x and the y for each category
        ar = df[df[category] == cat][xlabel]
        pop = df[df[category] == cat][ylabel]

        # plot the data
        ax.scatter(ar, pop, label = cat, s = 10)

    # ----------------------------------------------------------------------------------------------------
    # prettify the plot

    # eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant
    ax.spines["top"].set_color("None") 
    ax.spines["right"].set_color("None")

    # set a specific label for each axis
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)

    # change the lower limit of the plot, this will allow us to see the legend on the left
    ax.set_xlim(-0.01) 
    ax.set_title("Scatter plot of " + xlabel + " vs " + ylabel)
    ax.legend(loc = "upper right", fontsize = 10);
"""
s = final[(final['nombre_carrera'] == "INGENIERIA CIVIL-REDISEÑO") | (final['nombre_carrera'] == "INGENIERIA INDUSTRIAL-REDISEÑO")]
scatterplot(s.groupby(["nombre_carrera", "codigo_malla"],as_index = False).segMatTot.mean(),"nombre_carrera", "codigo_malla", "segMatTot" )
"""

# In[34]:


def densityplot(df,category, xlabel):
    fig = plt.figure(figsize = (10, 8))

    for cyl_ in df[category].unique():
        x = df[df[category] == cyl_][xlabel]
        sns.kdeplot(x, shade=True, label = "{}".format(cyl_))

    plt.title("Density Plot of segMatTot by codigo_carrera");
"""
s = final[(final['nombre_carrera'] == "INGENIERIA CIVIL-REDISEÑO") | (final['nombre_carrera'] == "INGENIERIA INDUSTRIAL-REDISEÑO")]
densityplot(s.groupby(["nombre_carrera", "codigo_malla"],as_index = False).segMatTot.mean(),"nombre_carrera",  "segMatTot" )
"""

# In[37]:


def denshistogram(df,category, xlabel):
    df[category].unique()
    fig = plt.figure(figsize = (10, 8))

    for class_ in df[category].unique():
        x = df[df[category] == class_][xlabel]
        sns.distplot(x, kde = True, label = "{}".format(class_))

    plt.title("Density Plot of segMatTot by codigo_carrera");
"""   
sh = final[(final['nombre_carrera'] == "INGENIERIA CIVIL-REDISEÑO") | (final['nombre_carrera'] == "INGENIERIA INDUSTRIAL-REDISEÑO")]
denshistogram(sh.groupby(["nombre_carrera", "codigo_malla"],as_index = False).segMatTot.mean(),"nombre_carrera",  "segMatTot" )
"""

# In[5]:


def categoricalplot(df, category, xlabel):
    fig = plt.figure(figsize = (12, 6))
    ax = sns.catplot(category, 
                     col = xlabel, 
                     data = df, 
                     kind = "count",  
                     palette = 'tab20',  
                     aspect = .8);
"""    
c = final[(final['nombre_carrera'] == "INGENIERIA CIVIL-REDISEÑO") | (final['nombre_carrera'] == "ARQUITECTURA")]
categoricalplot(c,"dropout",  "nombre_carrera" )
"""

# In[4]:


def piechart(df, category, title):
    d = df[category].value_counts().to_dict()
    fig = plt.figure(figsize = (18, 6))
    ax = fig.add_subplot()

    ax.pie(d.values(), 
            labels = d.keys(), 
            autopct = '%1.1f%%',
           textprops={'fontsize': 10}
          )

    ax.set_title("Pie chart")
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize = 10, title = title);
"""
p = final[(final['nombre_carrera'] == "INGENIERIA CIVIL-REDISEÑO")]
piechart(final,"nombre_carrera","Estado del alumno")
"""

# In[ ]:


def create_date_tick(df):
    '''
    Converts dates from this format: Timestamp('1949-01-01 00:00:00')
    To this format: 'Jan-1949'
    '''
    df["date"] = pd.to_datetime(df["date"]) # convert to datetime
    df["month_name"] = df["date"].dt.month_name() # extracts month_name
    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan
    df["year"] = df["date"].dt.year # extracts year
    df["new_date"] = df["month_name"].astype(str) + "-" +df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949

def timeseriesplot(df, category, label_, title):
    create_date_tick(df)

    y = df[category]

    max_peaks_index, _ = find_peaks(y, height=0) # find maximum using scipy library

    doublediff2 = np.diff(np.sign(np.diff(-1*y))) # find minimum using numpy library
    min_peaks_index = np.where(doublediff2 == -2)[0] + 1

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot()

    ax.plot(y, color = "blue", alpha = .5, label = label_)

    # we have the index of max and min, so we must index the values in order to plot them
    ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 90, color = "green", alpha = .5, label = "Peaks")
    ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 90, color = "red", alpha = .5, label = "Troughs")

    for max_annot, min_annot in zip(max_peaks_index[::3], min_peaks_index[1::5]):
        text = df.iloc[max_annot]["new_date"]

        ax.text(df.index[max_annot], y[max_annot] + 50, s = text, fontsize = 8, horizontalalignment='center', verticalalignment='center')
        ax.text(df.index[min_annot], y[min_annot] - 50, s = text, fontsize = 8, horizontalalignment='center', verticalalignment='center')

    xtick_location = df.index.tolist()[::6]
    xtick_labels = df["new_date"].tolist()[::6]

    ax.set_xticks(xtick_location)
    ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})

    ax.grid(axis = "y", alpha = .3)
    ax.set_ylim(0, 700)
    ax.set_title(title, fontsize = 14)
    ax.legend(loc = "upper left", fontsize = 10);
    


# In[ ]:


def timeseriesmultplot(df, category1, category2, datecolumn, title):
    df.set_index(datecolumn, inplace = True)

    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot()

    ax.plot(df[category1], color = "red", alpha = .5, label = category1)
    ax.plot(df[category2], color = "blue", alpha = .5, label = category2)
    ax.legend(loc = "upper left", fontsize = 10)
    ax.set_title(title, fontsize = 14)

    xtick_location = df.index.tolist()[::6]
    xtick_labels = df.index.tolist()[::6]

    ax.set_xticks(xtick_location)
    ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});

    ax.tick_params(axis='x', labelrotation = 90, labelsize = 12)
    ax.tick_params(axis='y', labelsize = 12)
    ax.grid(axis = "y", alpha = .3)


# In[28]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def autopartialcorrelation(df, category):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)

    plot_acf(df[category], ax = ax1, lags = 50)
    plot_pacf(df[category], ax = ax2, lags = 15);
"""
apc = final[(final['nombre_carrera'] == "ARQUITECTURA")] #& (final['dropout'] != 2)]
    
autopartialcorrelation(apc, "rateAprobadas")
"""

# In[20]:


from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
def clusterplot(df, xlabel, ylabel, index_to_cluster, title, ncluster):
    x = df[xlabel]
    y = df[ylabel]

    # first calculate 5 clusters
    cluster = AgglomerativeClustering(n_clusters = ncluster, 
                                      affinity = 'euclidean', 
                                      linkage = 'ward')  

    cluster.fit_predict(df[index_to_cluster])  

    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot()

    ax.scatter(x, y)

    # Encircle
    def encircle(x,y, ax = None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[x,y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices,:], **kw)
        ax.add_patch(poly)
    fc_ = ["gold", "tab:blue", "tab:green", "tab:orange", "tab:red"]
    # use our cluster fitted before to draw the clusters borders like we did at the beginning of the kernel
    for i in range(ncluster):
        encircle(df.loc[cluster.labels_ == i, xlabel], df.loc[cluster.labels_ == i, ylabel], ec="k", fc=fc_[i], alpha=0.2, linewidth=0)
   
    ax.tick_params("x", labelsize = 10)
    ax.tick_params("y", labelsize = 10)

    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)

    ax.set_title(title, fontsize = 14);
"""
index_to_cluster = ['rateAprobadas', 'mediaAPRP', 'terMatActual', 'segMatActual']
p = final[(final['nombre_carrera'] == "ARQUITECTURA")] #& (final['dropout'] != 2)]
print(p[["id", "rateAprobadas", "mediaAPRP", "porcentaje_carrera","dropout"]].sort_values(["rateAprobadas","mediaAPRP"], ascending = [False, True]).head())
clusterplot(p,"mediaAPRP", "rateAprobadas", index_to_cluster, "Clustering media APRP vs rateAprobadas", 5)
"""

# In[11]:


#For categorical
#p = final[(final['nombre_carrera'] == "ARQUITECTURA")] #& (final['dropout'] != 2)]

#(pd.crosstab([p["genero"],p["dropout"]], p["codigo_malla"],margins=True).style.background_gradient(cmap='summer_r'))
    
#crosstabplot(p, "genero", "dropout", "codigo_malla" )


# In[12]:


def factorplot(df, category1, category2, category3):
    sns.factorplot(category1,category2,hue=category3,data=df)
    plt.show()
"""
p = final[(final['nombre_carrera'] == "ARQUITECTURA")]
factorplot(p, "codigo_malla", "dropout", "genero")
"""

# In[18]:


def violinplot(df, category1, category2, category3, category4):
    f,ax=plt.subplots(1,2,figsize=(18,8))
    sns.violinplot(category1, category2, hue=category3, data=df,split=True,ax=ax[0])
    ax[0].set_title(category1 + ' and '+category2+' vs '+category3)
    ax[0].set_yticks(range(0,110,10))
    sns.violinplot(category4,category2, hue=category3, data=df,split=True,ax=ax[1])
    ax[1].set_title(category4 + ' and '+category2+' vs '+category3)
    ax[1].set_yticks(range(0,110,10))
    plt.show()
"""    
p = final[(final['nombre_carrera'] == "ARQUITECTURA") & (final['dropout'] != 2)]
violinplot(p, "total_asignaturas", "codigo_malla", "dropout", "genero")
"""

# In[20]:


def correlationheatmap(data):
    sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()
""" 
p = final[(final['nombre_carrera'] == "ARQUITECTURA") & (final['dropout'] != 2)]
correlationheatmap(p)
"""
