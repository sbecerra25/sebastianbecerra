#!/usr/bin/env python
# coding: utf-8

# # PREGUNTA 1

# In[62]:


import pandas as pd
import matplotlib as plt
import numpy as np

#Load the database
path = "C:/Users/sbece/Downloads/ML data/desafio_spike_cuencas-master/desafio_spike_cuencas-master/caudal_extra.csv"
datos = pd.read_csv(path, delimiter = ",")
datos.head(5)
datos = datos.drop('Unnamed: 0', axis = 1)
datos.head()


# In[63]:


datos.dtypes


# In[64]:


#Convert fecha to datetime and delete hours, minutes and seconds
from datetime import datetime

datos['fecha'] = pd.to_datetime(datos['fecha'])
#datos['fecha'] = pd.to_datetime(datos['fecha']).dt.date
print (datos['fecha'].head().apply(type))

#extract months and days
#datos['month'] = pd.DatetimeIndex(datos['fecha']).month
#datos['day'] = pd.DatetimeIndex(datos['fecha']).day
#datos.head()


# # PREGUNTA 2

# In[65]:


##Summary - statistics
datos.describe()


# In[66]:


#Get the categorical variables
describe = datos.describe()
cols = datos.columns
num_cols = datos._get_numeric_data().columns
print(num_cols) ##numerical variables
categorias = list(set(cols) - set(num_cols))
print(categorias)


# In[67]:


#Missing values
datos.isnull().values.any() #is there any NaN value?
datos.isnull() ## rows with na values
datos.isnull().sum() ##total number of missing values by variable


# In[68]:


##there are many missing values in the variables precip_promedio and temp_max_promedio
## let's take a look at those variables
datos["precip_promedio"].isnull()
datos["temp_max_promedio"].isnull()

##Podemos concluir que hay estaciones que no tienen estaciones para medir la temperatura y precipitación. Por ello, se decidió
##por simplicidad, eliminar estas observaciones.


# In[69]:


##drop missing values
datos = datos.dropna()
datos.isnull().sum() 


# In[70]:


#Histograms
##First of all, we must get all numerical variables
#num_cols = datos.columns[datos.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
num = datos[num_cols]
print(num.head())
##codigo_estacion should be a categorical variable or dummy variable so keep that in mind.


# In[71]:


num = num.drop(['codigo_estacion', 'codigo_cuenca', 'gauge_id'] , 1) 
num.hist()


# In[72]:


#Density plots
import seaborn as sns

sns.distplot(num['caudal'], hist = True, kde = True,
            bins = int(1000/50), color = 'blue',
            hist_kws = {'edgecolor' : 'black'},
            kde_kws = {'linewidth' : 4})
#Right-skewed distribution


# In[73]:


sns.distplot(num['precip_promedio'], hist = True, kde = True,
            bins = int(1000/50), color = 'blue',
            hist_kws = {'edgecolor' : 'black'},
            kde_kws = {'linewidth' : 4})
#Right-skewed distribution


# In[74]:


sns.distplot(num['temp_max_promedio'], hist = True, kde = True,
            bins = int(1000/50), color = 'blue',
            hist_kws = {'edgecolor' : 'black'},
            kde_kws = {'linewidth' : 4})

#Bimodal distribution


# In[17]:


#boxplot by variable
import matplotlib.pyplot as plt

#num[['caudal', 'precip_promedio', 'temp_max_promedio']].plot(kind='box')
num['caudal'].plot(kind='box')


# In[ ]:


num['precip_promedio'].plot(kind='box')
plt.show()


# In[ ]:


num['temp_max_promedio'].plot(kind='box')
plt.show()


# In[18]:


#correlation between variables
num[['caudal', 'precip_promedio', 'temp_max_promedio']].corr()

#Se podría pensar que no hay una relación de causalidad entre las variables porque sus correlaciones son bajísimas, es decir,
#no hay relación directa.


# # PREGUNTA 3

# In[75]:


import matplotlib.pyplot as plt

def time_plot_una_estacion(codigo_estacion, columna, fecha_min, fecha_max, colores):
    data = datos[datos["codigo_estacion"] == codigo_estacion]
    data = data[(data["fecha"] >= fecha_min) & (data['fecha'] <= fecha_max)]
    data.plot(x = 'fecha', y = columna, color = colores)
    plt.show() 
    
time_plot_una_estacion(7322001, 'caudal', '2010-02-03', '2017-01-01', colores = 'green')
time_plot_una_estacion(7322001, 'precip_promedio', '2010-02-03', '2017-01-01', colores = 'blue')
time_plot_una_estacion(7322001, 'temp_max_promedio', '2010-02-03', '2017-01-01', colores = 'orange')


# In[20]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

#, fecha_min, fecha_max
aux = []
def time_plot_estaciones_varias_columnas(codigo_estacion, columnas, fecha_min, fecha_max):
    data = datos[datos["codigo_estacion"] == codigo_estacion]
    data = data[(data["fecha"] >= fecha_min) & (data['fecha'] <= fecha_max)]
    for i in range(len(columnas)):
        aux0 = columnas[i] + '_normalized'
        aux.append(aux0)
        data[aux0] = float(data[columnas[i]] - data[columnas[i]].min)/(data[columnas[i]].max - data[columnas[i]].min)
    print(type(aux))  
    data.plot(x = 'fecha', y = aux)
    plt.show() 

time_plot_estaciones_varias_columnas(7322001, ['temp_max_promedio','caudal'], '2010-02-03', '2017-01-01')


# # PREGUNTA 4

# In[76]:


#extract years, months and days
datos['year'] = pd.DatetimeIndex(datos['fecha']).month
datos['month'] = pd.DatetimeIndex(datos['fecha']).month
datos['day'] = pd.DatetimeIndex(datos['fecha']).day
print(datos.head())

#https://datatofish.com/if-condition-in-pandas-dataframe/
def estaciones(row):
    if row['month'] <= 2:
        return 'Verano'
    elif row['month'] > 9 and row['month'] < 12:
        return 'Primavera'
    elif row['month'] >= 4 and row['month'] < 6:
        return 'Otoño'
    elif row['month'] >= 7 and row['month'] < 9:
        return 'Invierno'
    elif row['month'] == 3:
        if row['day'] <= 20:
            return 'Verano'
        else:
            return 'Otoño'
    elif row['month'] == 6:
        if row['day'] <= 21:
            return 'Otoño'
        else:
            return 'Invierno'
    elif row['month'] == 9:
        if row['day'] <= 23:
            return 'Invierno'
        else:
            return 'Primavera'
    elif row['month'] == 12:
        if row['day'] <= 21:
            return 'Primavera'
        else:
            return 'Verano'    
    else:
        return 'Error. Fecha incorrecta.'

datos['Estacion'] = datos.apply(estaciones, axis=1)
##number of observations by season
print(pd.value_counts(datos.Estacion))


# In[77]:


##Group the data by season
is_ver =  datos['Estacion'] == 'Verano' 
verano = datos[is_ver]
is_ot =  datos['Estacion'] == 'Otoño' 
otono = datos[is_ot]
is_inv =  datos['Estacion'] == 'Invierno' 
invierno = datos[is_inv]
is_pri =  datos['Estacion'] == 'Primavera' 
primavera = datos[is_pri]

def extremo(estacion):
    percentil1 = np.percentile(estacion['caudal'], 95)
    percentil2 = np.percentile(estacion['temp_max_promedio'], 95)
    percentil3 = np.percentile(estacion['precip_promedio'], 95)
    estacion.loc[estacion.caudal > percentil1, 'caudal_extremo'] = 1
    estacion.loc[estacion.caudal <= percentil1, 'caudal_extremo'] = 0
    estacion.loc[estacion.temp_max_promedio > percentil2, 'temp_extremo'] = 1
    estacion.loc[estacion.temp_max_promedio <= percentil2, 'temp_extremo'] = 0
    estacion.loc[estacion.precip_promedio > percentil3, 'precip_extremo'] = 1
    estacion.loc[estacion.precip_promedio <= percentil3, 'precip_extremo'] = 0

#databases with new values
extremo(verano)
extremo(otono)
extremo(invierno)
extremo(primavera)


# In[78]:


#concatenate the DataFrames
seasons = pd.concat([verano, otono, invierno, primavera])
seasons.head()


# # PREGUNTA 5

# In[25]:


#seasons[["codigo_estacion", "caudal_extremo"]].describe()

#seasons.groupby(['codigo_estacion'])['caudal_extremo'].describe()[['count', 'mean']]
#seasons.groupby(['codigo_estacion'])['caudal_extremo'].value_counts()

seasons1 = seasons[seasons['caudal_extremo'] == 1]
seasons1.groupby(['gauge_name'])['caudal_extremo'].value_counts()


##Se puede observar que la cantidad de caudal extremo por zona geográfica varia considerablemente, ya que por ejemplo si
##comparamos las cuencas del río Aysén en Puerto Aysén con el río Ancoa en El Morro (Linares) es garrafal, así como también
##lo es entre el río Biobio en Desembocadura y el río Choapa (Coquimbo).
##Tal vez sería interesante tener una variable que describa el clima donde se encuentra la cuenca para comparar valores
##extremos. 


# In[27]:


a = seasons1.groupby(['gauge_name'])['caudal_extremo'].value_counts()
b = seasons1['gauge_name'].value_counts()


# # PREGUNTA 6

# In[79]:


import matplotlib.pyplot as plt

seasons['fecha'] = pd.to_datetime(seasons['fecha']).dt.normalize()

b1 = seasons[seasons['caudal_extremo'] == 1]
a1 = b1.groupby(['fecha'])['caudal_extremo'].value_counts()
plt.figure(figsize=(20,5))
a1.plot(x ='fecha', y ='caudal_extremo', color = 'green') 
plt.show() 

b2 = seasons[seasons['temp_extremo'] == 1]
a2 = b2.groupby(['fecha'])['temp_extremo'].value_counts()
plt.figure(figsize=(20,5))
a2.plot(x ='fecha', y ='temp_extremo', color = 'red') 
plt.show()

b3 = seasons[seasons['precip_extremo'] == 1]
a3 = b3.groupby(['fecha'])['precip_extremo'].value_counts()
plt.figure(figsize=(20,5))
a3.plot(x ='fecha', y ='precip_extremo', color = 'orange') 
plt.show()

#Se puede apreciar que los eventos de caudal extremo se han hecho más comunes el último año, al igual que las temperaturas
#y precipitaciones extremas.


# # PREGUNTA 7

# In[30]:


##Dado que la variable caudal_extremo es binaria y las variables precip_promedio y temperatura_maxima promedio
##dependen del tiempo, no sería adecuado usar directamente un modelo de regresión logística.
seasons['caudal_extremo'].sum()
seasons['caudal_extremo'].value_counts()


# In[31]:


seasons['caudal_extremo'].sum()


# In[84]:


##A modo de ejemplo no se implementará un modelo Binary ARMA y se usarán otros algoritmos para predecir el caudal extremo.

##Antes de usar cualquier algoritmo es necesario transformar las variables categóricas a numéricas. Es por eso que se debe
## usar LabelEncoder o OneHotEnconder. Esta vez se usará OneHotEncoder porque tenemos más de dos categorías.
#Categorical boolean mask
#X = seasons.drop(['caudal_extremo', 'institucion' ,'fuente', 'nombre_sub_cuenca', 'gauge_name', 'nombre', 'fecha'], axis = 1)
#print(X.dtypes)
#y = seasons['caudal_extremo']
categorical_feature_mask = seasons.dtypes == object
#filter categorical columns using mask and turn it into a list
categorical_cols = seasons.columns[categorical_feature_mask].tolist()

#LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#apply le on categorical feature columns
seasons[categorical_cols] = seasons[categorical_cols].apply(lambda col: le.fit_transform(col))
print(seasons[categorical_cols].head())

#OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder

#ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse = False)
#seasons = ohe.fit_transform(seasons)
print(seasons.head())


# In[83]:


#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse = False)
seasons_ohe = ohe.fit_transform(seasons.drop(['fecha'], axis = 1))


# In[ ]:


##Train the model
# Split-out validation dataset
array = datos.values
print(array.shape)
#X = array[:, ]
#Y = array[:,4]
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[ ]:


import datetime

fecha_min = '2010-02-03'
#fecha_min = datetime.strptime(fecha_min, '%Y-%m-%d').date
type(fecha_min)
fecha_min = datetime.datetime.strptime(fecha_min, '%Y-%m-%d')
date_time_obj.day


# In[ ]:


#datos.loc[datos['fecha'] >=  '2010-02-03'  or datos['fecha'] <= '2017-01-01']
datos[(datos["fecha"] >= '2010-02-03') & (datos['fecha'] <= '2017-01-01')]


# In[36]:


seasons.head()


# In[53]:


import pandas as pd

pd.get_dummies(seasons['codigo_estacion'])

