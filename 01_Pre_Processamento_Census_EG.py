# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:48:39 2020

@author: egsil
"""
import pandas as pd 

#### BASE DO CENSO ####
base = pd.read_csv('census.csv')

##Transformação de variaveis categorica para numericas

#Dividir -- previsoes e classe

previsores = base.iloc[:,0:14].values

classe = base.iloc[:,14].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder()

#labels = labelencoder_previsores.fit_transform(previsores[:,1])

previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

#Classe

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


## Variavel Dummy - para controlar categoria (Peso) -- Criação de novas variaveis

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

#Escalonamento

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
