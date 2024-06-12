import streamlit as st
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import datetime
import pathlib
import plotly.express as px
import paquetes.modulo as md
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as ply
import sklearn
from pmdarima import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def quitar_dias(str):
    lista = str.split("/",2)
    return lista[2] + "-" + lista[0]

def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

def random_forest_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict([testX])
    return yhat[0]

def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

from sklearn.metrics import mean_absolute_error
def walk_forward_validation(data, n_test, algo):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        if algo == 0:
            yhat = random_forest_forecast(history, testX)
        else:
            yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = sqrt(mean_squared_error(test[:, -1], predictions))
    return error, test[:, -1], predictions

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def parametros(option):
    lista = []
    if option == "ARIMA":
        param_AR = st.number_input("Inserta el parámetro de AR (Modelo Autorregresivo)",min_value=0, step=1, placeholder="a1")
        lista.append(param_AR)
        param_I = st.number_input("Inserta el parámetro de I (Modelo Integrado)",min_value=0, step=1, placeholder="a2")
        lista.append(param_I)
        param_MA = st.number_input("Inserta el parámetro de MA (Modelo de Media Móvil)",min_value=0, step=1, placeholder="a3")
        lista.append(param_MA)
    if option == "SARIMAX":
        param_AR = st.number_input("Inserta el parámetro de AR (Modelo Autorregresivo)",min_value=0, step=1, placeholder="sa1")
        lista.append(param_AR)
        param_I = st.number_input("Inserta el parámetro de I (Modelo Integrado)",min_value=0, step=1, placeholder="sa2")
        lista.append(param_I)
        param_MA = st.number_input("Inserta el parámetro de MA (Modelo de Media Móvil)",min_value=0, step=1, placeholder="sa3")
        lista.append(param_MA)
        param_S = st.number_input("Inserta el parámetro de S",min_value=2, step=1, placeholder="sa4")
        lista.append(param_S)
    return lista
        
@st.cache_data
def load_data():
    dataframe = pd.read_csv("resources\\car_data.csv")
    dataframe["ini_month"] = dataframe["Date"].map(quitar_dias) + "-1"
    #print(dataframe["ini_month"])
    #dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe["ini_month"] = pd.to_datetime(dataframe["ini_month"])
    dataframe.sort_values("ini_month")
    agrupacion = dataframe.groupby("ini_month").count()
    agruped = agrupacion["Car_id"]
    return agruped

st.set_page_config(layout="wide")

md.menu()

option = st.selectbox(
   "Seleccione el modelo a usar",
   options = ("None","ARIMA","SARIMAX","RANDOM FOREST","XGBOOST"),
   index=None,
   placeholder="Modelo",
)
test = st.number_input("Seleccione el número de muestras para el test (1-24)",min_value=1,max_value=24, step=1, placeholder="a1")
data = load_data()
if option == "ARIMA":   
    l = parametros(option)
    if st.button("mostrar resumen del modelo"):
        model = ARIMA(data[:19], order=(l[0],l[1],l[2]))
        model_fit = model.fit()

        st.write(model_fit.summary())

        residuals = pd.DataFrame(model_fit.resid)

        st.write()

        chart=alt.Chart(residuals).mark_line()
        st.plotly_chart(chart)

        plot = residuals.plot(kind='kde')
        st.write(plot)

if option == "SARIMAX":
    l = parametros(option)
    warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
    modelo = SARIMAX(endog = data, order = (l[0], l[1], l[2]), seasonal_order = (l[0], l[1], l[2], l[3]))
    modelo_res = modelo.fit(disp=0)
    warnings.filterwarnings("default")
    if st.button("Mostrar resumen del modelo"):
        st.write(modelo_res.summary())

    if st.button("Mostrar predicciones del modelo"):
        predicciones_statsmodels = modelo_res.get_forecast(12).predicted_mean
        predicciones_statsmodels.name = 'predicciones_statsmodels'
    
        total = pd.concat([data,predicciones_statsmodels.head(12)], axis= 1)
        st.line_chart(total)

if option == "RANDOM FOREST":
    lista_agrupados = list(data.values)
    lista_procesada = series_to_supervised(lista_agrupados)
    error, tested, predicciones = walk_forward_validation(lista_procesada,test,0)

    data.columns = ["vendidos"]
    indices = data.index.values
    df_tested = pd.DataFrame(tested, index = indices[24-test:], columns=["testeados"])
    df_predicciones = pd.DataFrame(predicciones, index = indices[24-test:], columns=["predicciones"])
    total = pd.concat([data[:24-test],df_tested, df_predicciones], axis = 1)

    st.line_chart(total)
    st.write("Error cuadrático medio: " + str(round(error,2)))

if option == "XGBOOST":
    lista_agrupados = list(data.values)
    lista_procesada = series_to_supervised(lista_agrupados)
    error, tested, predicciones = walk_forward_validation(lista_procesada,test,1)

    indices = data.index.values
    df_tested = pd.DataFrame(tested, index = indices[24-test:], columns=["testeados"])
    df_predicciones = pd.DataFrame(predicciones, index = indices[24-test:], columns=["predicciones"])
    total = pd.concat([data[:24-test],df_tested, df_predicciones], axis = 1)

    st.line_chart(total)
    st.write("Error cuadrático medio: " + str(round(error,2)))
