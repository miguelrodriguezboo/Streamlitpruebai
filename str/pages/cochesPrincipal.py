from datetime import datetime
import streamlit as st
import pandas as pd
import time
import paquetes.modulo as md
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def load_data():
    data = pd.read_csv('resources\\car_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

with open('.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

if st.session_state["authentication_status"]:
    authenticator.logout(location="sidebar")
    st.title('Sales Cars')
    md.menu()
    data = load_data()

    data['year'] = data['Date'].dt.year

    #filtros
    all_years = sorted(data['year'].unique())
    selected_years = st.multiselect('Select Year:', all_years, default=all_years)

    all_regions = sorted(data['Dealer_Region'].unique())
    selected_regions = st.multiselect('Select Dealer Region:', all_regions, default=all_regions)

    filtered_data = data[data['year'].isin(selected_years) & data['Dealer_Region'].isin(selected_regions)]

    #datas enseñar
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    if st.checkbox('Show filtered data'):
        st.subheader('Filtered Data')
        st.write(filtered_data)

 
    #autos vendidos mes actual
    current_month = datetime.now().month 
    current_year = datetime.now().year -1

    current_month_data = data[(data['Date'].dt.month == current_month) & (data['Date'].dt.year == current_year)]

    totalAutomoviles = current_month_data['Car_id'].count()
    totalRevenue =current_month_data['Price ($)'].sum()
    revenueCOche = totalRevenue/totalAutomoviles

    totalRevenueM = totalRevenue / 1_000_000  
    FtotalRevenue = f"{totalRevenueM:.2f} M"  


    # Obtener los datos del mes anterior
    previous_month = current_month - 1
    previous_year = current_year
    if previous_month == 0:
        previous_month = 12
        previous_year -= 1

    previous_month_data = data[(data['Date'].dt.month == previous_month) & (data['Date'].dt.year == previous_year)]

    # Calcular los valores del mes anterior
    previous_totalAutomoviles = previous_month_data['Car_id'].count()
    previous_totalRevenue = previous_month_data['Price ($)'].sum()
    previous_revenueCOche = previous_totalRevenue / previous_totalAutomoviles
    previous_totalRevenueM = previous_totalRevenue / 1_000_000
    previous_FtotalRevenue = f"{previous_totalRevenueM:.2f} M"

    # Calcular los porcentajes respecto al mes anterior
    autos_porcentaje = ((totalAutomoviles / previous_totalAutomoviles) - 1) * 100
    revenue_porcentaje = ((totalRevenue / previous_totalRevenue) - 1) * 100
    revenue_por_coche_porcentaje = ((revenueCOche / previous_revenueCOche) - 1) * 100

    # Mostrar métricas con porcentaje respecto al mes anterior
    st.header("KPI Metrics (vs Mes Anterior)")

    row = st.columns(3)

    # Autos vendidos
    with row[0]:
        st.metric(label="Autos Vendidos", value=totalAutomoviles, delta=f"{autos_porcentaje:.2f}%")

    # Revenue
    with row[1]:
        st.metric(label="Revenue", value=f"${totalRevenue:,}", delta=f"{revenue_porcentaje:.2f}%")

    # Revenue por coche
    with row[2]:
        st.metric(label="Revenue por Coche", value=f"${FtotalRevenue}", delta=f"{revenue_por_coche_porcentaje:.2f}%")

    # company
    sales_summary = filtered_data.groupby('Company').agg(
        total_sales=('Price ($)', 'sum'),
        total_cars_sold=('Car_id', 'count')
    ).reset_index()

    sales_summary['revenue_per_car'] = sales_summary['total_sales'] / sales_summary['total_cars_sold']

    sales_summary['link'] = "http://localhost:8501/Marcas?marca=" + sales_summary['Company']
    #st.write("Summary by Company:", sales_summary)
    st.data_editor(
        sales_summary,
        column_config={
            "link": st.column_config.LinkColumn(
                "Enlace a detalles",
                help="Enlace a la página de detalles de la marca",
                #validate="^https://[a-z]+\.streamlit\.app$",
                display_text = "Enlace a detalle",
                max_chars=100,
            ),
        },
        hide_index=True,
    )

    #date
    aggregated_data = filtered_data.groupby('Date').agg(
        total_cars_sold=('Car_id', 'count'),
        total_revenue=('Price ($)', 'sum')
    ).reset_index()

    aggregated_data['revenue_per_car'] = aggregated_data['total_revenue'] / aggregated_data['total_cars_sold']

    option = st.selectbox(
        'Select the metric to display:',
        ('Automóviles vendidos', 'Revenue Total', 'Revenue por coche')
    )

    if option == 'Automóviles vendidos':
        st.write("Automóviles Vendidos por Fecha")
        chart_data = aggregated_data[['Date', 'total_cars_sold']].set_index('Date')
    elif option == 'Revenue Total':
        st.write("Revenue Total por Fecha")
        chart_data = aggregated_data[['Date', 'total_revenue']].set_index('Date')
    elif option == 'Revenue por coche':
        st.write("Revenue por Coche por Fecha")
        chart_data = aggregated_data[['Date', 'revenue_per_car']].set_index('Date')

    st.line_chart(chart_data)
    # marca con mas automoviles vendidos
    company = filtered_data.groupby('Company')
    ventasCompany = company['Car_id'].count()
    CompañiaMax = ventasCompany.idxmax()

    # Mostrar métricas
    st.header("Métricas de la Compañía con Más Ventas")
    row3 = st.columns(3)
    with row3[1]:
        st.metric(label="Compañía con más ventas :", value=CompañiaMax)

    # Calcular y mostrar autos vendidos y revenue total de la compañía con más ventas
    ventas_max = company.get_group(CompañiaMax)
    totalAutosMax = ventas_max['Car_id'].count()
    totalRevenueMax = ventas_max['Price ($)'].sum()


    row2 = st.columns(3)

    with row2[0]:
        st.metric(label="Autos Vendidos (Compañía Máx.)", value=totalAutosMax)
    with row2[2]:
        st.metric(label="Revenue Total (Compañía Máx.)", value=f"${totalRevenueMax:,.2f}")

elif st.session_state["authentication_status"] is False:
    authenticator.login()
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    authenticator.login()
    st.warning('Please enter your username and password')



