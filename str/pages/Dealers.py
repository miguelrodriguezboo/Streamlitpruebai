import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import datetime
import pathlib
import plotly.express as px
import paquetes.modulo as md
from typing import List, Tuple
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def number_to_month(numero_mes):
    d=['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto',
       'Septiembre','Octubre','Noviembre','Diciembre']
    return d[numero_mes - 1]

#calcula en nivel de ingreso dado el ingreso anual
def ingreso_anual(data):
    if (data > 1000000):
        return "Muy Alto"
    elif (data > 750000):
        return "Alto"
    elif (data > 500000):
        return "Medio-Alto"
    elif (data > 250000):
        return "Medio"
    elif (data > 100000):
        return "Medio-Bajo"
    return "Bajo"

#crea una gráfica circular de porciones
def pie_chart(data,agrupacion,agrupar,tam,colores):
    agrupacion_parcial = data.groupby(agrupacion)[agrupar].sum()
    agrupacion_total = data[agrupar].sum()
    etiquetas = agrupacion_parcial.index.to_list()

    tam_porciones = []
    for i in etiquetas:
        tam_porciones.append(agrupacion_parcial[i]/agrupacion_total)

    #creacion del gráfico circular
    fig1, ax1 = plt.subplots(figsize=(tam[0],tam[1]))
    ax1.pie(tam_porciones, labels = etiquetas,radius = 0.2, autopct="%1.1f%%", startangle= 90, colors = colores)
    ax1.axis('equal')
    return fig1

def pie_chart_2(data, agrupacion, agrupar,colores, tam, leyenda):
    agrupado = pd.Series.to_frame(data.groupby(agrupacion)[agrupar].sum())
    nombre_nueva_col = "c " + agrupacion
    agrupado[nombre_nueva_col] = agrupado.index.values

    fig = px.pie(agrupado,values =agrupar, names=nombre_nueva_col,color_discrete_sequence=colores)
    fig.update_layout(
        autosize= False,
        width=tam,
        height=tam,
        showlegend = leyenda
    )
    return fig

def fun_num(num):
    if (num < 10):
        return "0" + str(num)
    return str(num)

def str_to_MY(cadena : str):
    meses =['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto',
       'Septiembre','Octubre','Noviembre','Diciembre']
    mes = int(cadena[-2:])
    return meses[mes - 1] + " " + cadena[:4]
def work_directory():
    return str(pathlib.Path(__file__).parent.absolute().parent.absolute())

@st.cache_data
#cargamos datos del csv
def load_data():
    dir = work_directory()
    data = pd.read_csv(dir + "\\resources\\car_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    #data['M/Y'] = data['Month'].map(number_to_month) + " " + data['Year'].map(str)
    data['M/Y'] = data['Year'].map(str) + data['Month'].map(fun_num)
    data["Ingreso Anual"] = data["Annual Income"].apply(lambda x: ingreso_anual(x))
    data = data.sort_values(by=['Year','Month'])
    return data

#funcion para desplegar los kpis
def display_kpi_metrics(kpis: List[float], kpi_names: List[str], delta):
    st.header("KPI Metrics")
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(3), zip(kpi_names, kpis))):
        col.metric(label=kpi_name, value=str(kpi_value)+"%", delta=delta[i])

def car_sold(data, year, month):
    data_y = data[data['Year'] == (data['Year'].max() - year)]
    data_m = data_y[data['Month'] == (data['Month'].max() - month)]
    data_d = data_m[data['Dealer_Name'] == dealer]
    return data_d

def porcentage_relationship(value1, value2, decimals):
    total = ((value1 / value2) - 1) * 100
    if(decimals != -1):
        total = round(total,decimals)
    return total

def lista_to_MY(lista):
    nuevo_indice = []
    for i in lista:
        nuevo_indice.append(str_to_MY(i))
    return nuevo_indice

#devuelve el diccionario con los valores entre start_value y end_value
def cortar_diccionario(start_value, end_value, diccionario):
    nuevo_diccionario = {}
    lista_claves = list(diccionario.keys())
    tam_lista = len(lista_claves)
    i = False
    aux = 0

    while (aux < tam_lista):
        clave = lista_claves[aux]

        if clave == start_value:
            i = True
        if i:
            nuevo_diccionario[clave] = diccionario[clave]     
        if clave == end_value:
            i = False

        aux+=1
    
    return nuevo_diccionario

#crea gráfica multilíneas pasandole dataframe, nombre de las variables a mostrar y los colores para representarlas
def multi_grafica(data, etiquetas, colores):
    
    graf = alt.Chart(data).mark_line().encode(
        x=alt.X('M/Y', sort=None),
        y='value',
        color=alt.Color('variable', scale=alt.Scale(domain=etiquetas, range=colores))
    )
    return graf

def cortar_lista(start_value, end_value, lista):
    nueva_lista = []
    tam_lista = len(lista)
    i = False
    aux = 0

    while (aux < tam_lista):
        clave = lista[aux]
        if clave == start_value:
            i = True
        if i:
            nueva_lista.append(clave)
        if clave == end_value:
            i = False
        aux+=1
    return nueva_lista
#establece la página a ventana completa
st.set_page_config(layout="wide")

with open('str/.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)
nombre, estado, username = authenticator.login()

if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = estado
if 'nombre' not in st.session_state:
    st.session_state.nombre = nombre

if st.session_state["authentication_status"]:    
    authenticator.logout(location="sidebar")
    md.menu()
    st.title('Dealers')
    data = load_data()
    columnas = data.columns.values
    container = st.container(border=True)
    dealer = st.selectbox('Select Dealer:', data['Dealer_Name'].unique())
    num_dealers = len(data['Dealer_Name'].unique())


    ##creación de las metricas/KPIs
    car_actual_month = car_sold(data,0,0)
    car_previous_month = car_sold(data,0,1)

    #automóviles vendidos mes actual y mes previo - KPI1
    autos_vendidos_actual = car_actual_month['Car_id'].count()
    autos_vendidos_previous = car_previous_month['Car_id'].count()
    autos_vendidos_total = int(autos_vendidos_actual - autos_vendidos_previous)

    #revenue obtenido el mes actual y el mes anterior - KP2
    autos_revenue_actual = car_actual_month['Price ($)'].sum()
    autos_revenue_previous = car_previous_month['Price ($)'].sum()
    autos_revenue_total = int(autos_revenue_actual - autos_revenue_previous)

    #revenue promedio obtenido por coche el mes actual y el mes anterior - KPI3
    revenue_por_coche_actual = autos_revenue_actual / autos_vendidos_actual
    revenue_por_coche_previous = autos_revenue_previous / autos_vendidos_previous
    revenue_por_coche_total = float(round(revenue_por_coche_actual - revenue_por_coche_previous,2))

    #porcentajes de las 3 metricas
    porcentaje = porcentage_relationship(autos_vendidos_actual,autos_vendidos_previous,2)
    porcentaje2 = porcentage_relationship(autos_revenue_actual,autos_revenue_previous,2)
    porcentaje3 = porcentage_relationship(revenue_por_coche_actual, revenue_por_coche_previous,2)

    #Se muestran los KPIs
    kpi_names = ["Autos vendidos","Revenue","Revenue por coche"]
    kpis = [porcentaje, porcentaje2, porcentaje3]
    deltas = [autos_vendidos_total, autos_revenue_total, revenue_por_coche_total]
    display_kpi_metrics(kpis, kpi_names, deltas)

    #escogemos los datos del dealer seleccionado
    data_dealer = data[data['Dealer_Name'] == dealer]
    lista = list(data_dealer['M/Y'].unique())
    lista = list(map(str_to_MY, lista))

    #botón para poder obtener el fichero csv
    flag = st.button("Obtener datos en formato CSV vía correo electrónico", type="primary")

    archivo_csv = work_directory() + "\\resources\\dealers.csv"

    #inicializamos la variable placeholder
    if 'placeholder' not in st.session_state:
        st.session_state.placeholder = 'formato: correo@dominio'
    if 'botoncito' not in st.session_state:
        st.session_state.botoncito = False

    nombre_del_archivo = "dealers.csv"
    correo = ""

    if flag:
        data_dealer.to_csv(archivo_csv, index=False)
        st.button("Enviar a mi correo", key="botoncito")
        correo = st.text_input(
                            'Escriba el correo electrónico al que desea enviar los datos',
                            "formato: correo@dominio",
                                key='placeholder',
                            )
        
    if st.session_state["botoncito"]:
        correo = config['credentials']['usernames'][st.session_state["username"]]['email']
    else:
        correo = st.session_state.placeholder

    if correo != "formato: correo@dominio":
        md.send_email(archivo_csv, nombre_del_archivo,correo)

    #reiniciamos el placeholder
    #st.session_state['placeholder'] = "formato: correo@dominio"
    
    #slider meses
    start, end = st.select_slider("Selecciona el rango de meses", options = lista, value=('Enero 2022','Diciembre 2023'))

    data_clients = data_dealer.groupby(['M/Y']).Phone.nunique()

    lista_valores = data_clients.values

    indice = data_clients.index
    nuevo_indice = lista_to_MY(indice)

    diccionario_valores = cortar_diccionario(start, end, dict(zip(nuevo_indice, lista_valores)))
    pandas_dic = pd.DataFrame(list(diccionario_valores.items()), columns=["Meses", "Vendidos"])

    dic ={'left': 5, 'top': 5, 'right': 5, 'bottom': 5}
    grafica = alt.Chart(pandas_dic, padding=dic).mark_bar(align="center", color="#000000", height = 0.5).encode(
        x=alt.X('Meses', sort=None),
        y='Vendidos',
    )

    #seleccion de los meses
    lista_cortada = cortar_lista(start, end, nuevo_indice)

    data_vehiculos = pd.Series(list(data_dealer.groupby(['M/Y']).Car_id.count()))
    data_vehiculos_total = pd.Series(list(data.groupby(['M/Y']).Car_id.count() / num_dealers))
    nuevo_indice = pd.Series(nuevo_indice)

    #para poder mostrar las 2 líneas en la gráfica
    nombre1 = "Promedio de ventas"
    data_vehiculos_total.name= nombre1
    nombre2 = "Ventas de " + dealer
    data_vehiculos.name = nombre2
    nombre3 = 'M/Y'
    nuevo_indice.name = nombre3
    lista_data = pd.concat([nuevo_indice,data_vehiculos,data_vehiculos_total], axis=1)

    #nos quedamos solo con los datos de los meses seleccionados en el slider
    nuevo_data = lista_data[lista_data['M/Y'].isin(lista_cortada)]

    #nombres de las columnas del dataframe y colores para cada uno de ellos
    nombres = [nombre1, nombre2]
    range = ["#E1C233","#000000"]
    #pivotamos la tabla con melt
    data = pd.melt(nuevo_data, id_vars='M/Y')
    graf = multi_grafica(data, nombres, range)

    h = 0
    row2 = st.columns(2)
    #representamos las gráficas en la misma fila
    for col in row2:
        if(h == 0):
            tile1 = col.container(height = 350, border=False)
            tile1.write(grafica)
        else:
            tile2 = col.container(height = 350, border=False)
            tile2.altair_chart(graf)
        h+=1

    autos = pd.Series.to_frame(data_dealer.groupby('Company')['Price ($)'].sum())
    top_5_autos = autos.nlargest(5,'Price ($)')

    row = st.columns(3)
    colores = ["#000000","#E1C233","#1FC2C2","#A666B0","#573B92","#666666"]
    colores2 = ["#000000","#E1C233"]
    #graf1 = pie_chart(data_dealer,'Ingreso Anual','Price ($)',[5,5], colores)
    graf1 = md.pie_chart_figura(data_dealer,'Ingreso Anual','Price ($)', colores,350, False,"suma")
    #graf2 = pie_chart(data_dealer,'Gender','Price ($)',[6,5],colores2)
    graf2 = md.pie_chart_figura(data_dealer,'Gender','Price ($)',colores2,350, False,"suma")
    lista = [graf1,graf2,top_5_autos]

    #representamos los gráficos de la segunda línea
    i = 0
    for col in row:
        tile = col.container(height=350, border=False)
        if( i== 2):
            tile.bar_chart(lista[i], color="#000000")
        else:
            tile.plotly_chart(lista[i])
        i+=1
    row3 = st.columns(2)
    t1 = row3[0].container(height=300, border= False)
    s = t1.selectbox("Compañias",options = top_5_autos.index)
    t2 = row3[1].container(height=350, border= False)
    t2.link_button("Ir a detalles de "+s , "http://localhost:8501/Marcas?marca="+s, use_container_width=True)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')