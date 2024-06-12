import plotly.express as px
import pandas as pd
import smtplib
import pathlib
import mimetypes
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

def pie_chart_figura(data, agrupacion, agrupar,colores, tam, leyenda,funcion):
    if funcion == "suma":
        agrupado = pd.Series.to_frame(data.groupby(agrupacion)[agrupar].sum())
    elif funcion == "size":
        agrupado = pd.Series.to_frame(data.groupby(agrupacion).size())
        agrupado.columns = [agrupar]

    nombre_nueva_col = "c " + agrupacion
    agrupado[nombre_nueva_col] = agrupado.index.values

    fig = px.pie(agrupado,values=agrupar, names=nombre_nueva_col, color=nombre_nueva_col,color_discrete_sequence=colores)
    fig.update_layout(
        autosize= False,
        width=tam,
        height=tam,
        showlegend = leyenda
    )
    return fig

def menu():
    #Muestra el nuevo menú
    st.sidebar.page_link("aplicacion.py", label="Inicio")
    st.sidebar.page_link("pages/cochesPrincipal.py", label="Datos generales")
    st.sidebar.page_link("pages/Dealers.py", label="Dealers")
    st.sidebar.page_link("pages/modelos.py", label="Modelos")
    st.sidebar.page_link("pages/modelos_rc.py", label="Modelos R/C")
    st.sidebar.page_link("pages/RedesN.py", label="Redes Neuronales")

def send_email(ruta_del_archivo, nombre_del_fichero, destinatario):
    direccion_origen = "streamlit11@gmail.com"
    password = "xvun alej lmbw poye"
    direccion_destino = destinatario
    fichero_a_mandar = ruta_del_archivo

    ctype, encoding = mimetypes.guess_type(fichero_a_mandar)

    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"

    maintype, subtype = ctype.split("/", 1)

    fp = open(fichero_a_mandar, "rb")
    attachment = MIMEBase(maintype, subtype)
    attachment.set_payload(fp.read())
    fp.close()
    encoders.encode_base64(attachment)

    attachment.add_header("Content-Disposition", "attachment", filename=nombre_del_fichero)


    message = MIMEMultipart()
    message["From"] = direccion_origen
    message["To"] = direccion_destino
    message["Subject"] = "Fichero de Dealer"
    message.attach(attachment)

    session = smtplib.SMTP("smtp.gmail.com", 587)
    session.starttls()
    session.login(direccion_origen, password)
    text = message.as_string()
    session.sendmail(direccion_origen, direccion_destino, message.as_string())
    session.quit()