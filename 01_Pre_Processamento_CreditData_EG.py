# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:02:28 2020

@author: egsil
"""

import pandas as pd

# Ler CSV - cria um dataframe
base = pd.read_csv('credit_data.csv')

#estatisticas
base.describe()

##Tratamento dos dados##

#Idades menor que <0
base.loc[base['age'] <0 ]

# Apagar a coluna
base.drop('age',1,inplace=True)

#Apagar somente registros com problema
base.drop(base[base.age <0].index,inplace=True)

#preencher os valores manualmente
#preencher os valores com a média
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age<0,'age'] = base['age'][base.age>0].mean()

##Valores Faltantes -- Dados Nulos##
pd.isnull(base['age'])
#localiza null no panda
base.loc[pd.isnull(base['age'])]

#separar 
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

#estrategia para preencimento de faltantes

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer = imputer.fit(previsores[:,0:3])

previsores[:,0:3]= imputer.transform(previsores[:,0:3])
#base.loc[pd.isnull(previsores[1])]

#Escalonamento - Deixar atributos na mesma escala
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Aplica a padronização
previsores = scaler.fit_transform(previsores)



