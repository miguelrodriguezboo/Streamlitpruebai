import streamlit_authenticator as stauth
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import paquetes.modulo as md
import base64

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://static.vecteezy.com/system/resources/previews/000/623/004/non_2x/auto-car-logo-template-vector-icon.jpg");
  background-size: 90%;
}
</style>
"""
st.set_page_config(layout="wide")
st.markdown(page_element, unsafe_allow_html=True)


with open('str/.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

if st.session_state.authentication_status:
    
    authenticator.logout(location="sidebar")
    md.menu()
    st.title(f':grey[Bienvenido *{st.session_state["name"]}*]')
else:
    st.title("Bienvenido")
    st.link_button("Iniciar Sesión", "https://dominiodeprueba/Login")
    st.link_button("Registrarse", "https://dominiodeprueba/Register")
