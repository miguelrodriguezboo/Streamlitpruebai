import streamlit as st
import warnings
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings
import paquetes.modulo as md
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('once')


def comprobar_errores(lista1,lista2):
    errores = 0
    for i in range(len(lista1)):
        if lista1[i] != lista2[i]:
            errores+=1
    return errores

def divisor(x):
    if x != 0:
        x = x/1000
    return x

def codificador(data, columna, codes, rango):
    for i in range(codes):
        data.loc[(data[columna] <= rango[i+1]) & (data[columna] >= rango[i]), columna] = i
    return data

def obtener_mejor_rmse(data, test_size, max_depth, rango):
    aux_rmse = 1000
    semilla = 0
    aux_modelo = None
    aux_predicciones = None
    aux_ytest = None

    for i in range(rango):
        X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(columns = "salary"),
                                        data['salary'],
                                        test_size=test_size,
                                        random_state = i
                                    )

        modelo = DecisionTreeRegressor(
            max_depth         = max_depth,
            random_state      = i
            )

        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X = X_test)

        rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
        )

        if rmse < aux_rmse:
            semilla = i
            aux_rmse = rmse
            aux_modelo = modelo
            aux_predicciones = predicciones
            aux_ytest = y_test
    
    return [semilla, aux_rmse, aux_modelo,aux_predicciones,aux_ytest]


@st.cache_data
def load_data():
    data = pd.read_csv("C:\\Users\\mrboo.INDRA\\Placement_Data_Full_Class.csv")
    return data

def clasificacion_alg(algoritmo, data):
    datos = transformar_datos(data,1)

    X_train, X_test, y_train, y_test = train_test_split(
                                        datos.drop(columns = "gender"),
                                        data['gender'],
                                        test_size=0.15,
                                        random_state = 431
                                    )
    
    clf = algoritmo
    clf.fit(X_train, y_train)

    lista = clf.predict(X_test)
    listilla = list(y_test)

    errores = comprobar_errores(lista, listilla)
    porcentaje_errores = errores/len(lista)

    df_l1 = pd.DataFrame(lista, columns=["Predecidos"])
    df_l2 = pd.DataFrame(listilla, columns=["Testeados"])

    df_total = pd.concat([df_l1,df_l2], axis=1)

    return porcentaje_errores, df_total


def transformar_datos(data,clasificacion):
    data['hsc_b']= data['hsc_b'].map( {'Others': 0, 'Central': 1} ).astype(int)
    data['gender']= data['gender'].map( {'M': 0, 'F': 1} ).astype(int)
    data['ssc_b']= data['ssc_b'].map( {'Others': 0, 'Central': 1} ).astype(int)
    data['hsc_s']= data['hsc_s'].map( {'Commerce': 0, 'Science': 1, 'Arts': 2} ).astype(int)
    data['degree_t']= data['degree_t'].map( {'Sci&Tech': 0, 'Comm&Mgmt': 1, 'Others': 2} ).astype(int)
    data['workex']= data['workex'].map( {'No': 0, 'Yes': 1} ).astype(int)
    data['specialisation']= data['specialisation'].map( {'Mkt&HR': 0, 'Mkt&Fin': 1} ).astype(int)
    data['status']= data['status'].map( {'Placed': 0, 'Not Placed': 1} ).astype(int)

    data["salary"]= data["salary"].fillna(0)
    data["salary"] = data["salary"].apply(divisor)

    l50_to_100 = [50,60,70,80,90,100]
    l14 = [30,44,58,72,86,100]

    if clasificacion:
        data.loc[data['ssc_p'] <= 50, 'ssc_p'] = 4
        data = codificador(data,'ssc_p',4,l50_to_100[:5])
        data = codificador(data,'hsc_p',5,l14)
        data = codificador(data,'degree_p',5,l50_to_100)
        data = codificador(data,'etest_p',5,l50_to_100)
        data = codificador(data,'mba_p',3,l50_to_100[:4])

    return data

md.menu()
data = load_data()

option = st.selectbox(
   "Seleccione el modelo a usar",
   options = ("None","ARBOL DECISION R","NAIVE BAYES","ARBOL DE DECISIÓN C","RANDOM FOREST","REDES NEURONALES"),
   index=None,
   placeholder="Modelo",
)

if(option == "ARBOL DECISION R"):
    datos = transformar_datos(data,0)

    #options = st.multiselect(
    #'Eliminar predictores:',
    #["sl_no","gender","ssc_p","ssc_b","hsc_b","hsc_s","degree_p",
    # "workex","etest_p","specialisation","degree_t","hsc_p","status","mba_p"])

    #datos = datos.drop(columns=options)
    
    lista = obtener_mejor_rmse(datos, 0.20, 4, 500)
    fig, ax = plt.subplots(figsize=(12, 5))

    modelo = lista[2]
    importancia_predictores = pd.DataFrame(
                            {'predictor': datos.drop(columns = "salary").columns,
                             'importancia': modelo.feature_importances_}
                            )
    st.write("Importancia de los predictores en el modelo:")
    st.write(importancia_predictores.sort_values('importancia', ascending=False))

    plot = plot_tree(
            decision_tree = lista[2],
            feature_names = data.drop(columns = "salary").columns,
            class_names   = ['salary'],
            filled        = True,
            impurity      = False,
            fontsize      = 10,
            precision     = 2,
            ax            = ax
        )
    
    df_predicted = pd.DataFrame(lista[3], columns=["predecidos"])
    df_tested = pd.DataFrame(list(lista[4]), columns=["testeados"])
    df_total = pd.concat([df_predicted,df_tested], axis=1)

    st.write("Error cuadrático medio: ", lista[1])
    st.write("Salarios empleados para el test vs Salario predecido")
    st.line_chart(df_total)
    st.write("Árbol de decision: ")
    st.pyplot(fig)

if option == "NAIVE BAYES":   
    errores, df_total = clasificacion_alg(GaussianNB(), data)    
    st.write("Porcentaje de errores en la predicción: ", round(errores*100,2), "%")
    st.line_chart(df_total)

if option == "ARBOL DE DECISIÓN C":
    errores, df_total = clasificacion_alg(DecisionTreeClassifier(), data)
    st.write("Porcentaje de errores en la predicción: ", round(errores*100,2), "%")
    st.line_chart(df_total)

if option == "RANDOM FOREST":
    errores, df_total = clasificacion_alg(RandomForestClassifier(), data)
    st.write("Porcentaje de errores en la predicción: ", round(errores*100,2), "%")
    st.line_chart(df_total)

if option == "REDES NEURONALES":
    st.write("Cargando")