import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
import geopandas
import plotly.io as pio
import joblib
import pickle
import altair as alt


df = pd.read_csv('Data.csv')
dt = pd.read_csv('Data_Cleaned.csv')
         
def plot_average_electricity():
    average_electricity_comp_by_year = df.groupby('Year')[['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)']].mean()
    average_electricity_comp_by_year = average_electricity_comp_by_year.reset_index()
    average_electricity_comp_by_year = average_electricity_comp_by_year.round(2)
    average_electricity_comp_by_year.set_index('Year', inplace=True)
    fig = go.Figure()

    for column in average_electricity_comp_by_year.columns:
        if column == 'Electricity from fossil fuels (TWh)':
            legend_label = 'Fossil Fuels'
        elif column == 'Electricity from nuclear (TWh)':
            legend_label = 'Nuclear'
        elif column == 'Electricity from renewables (TWh)':
            legend_label = 'Renewables'
        else:
            legend_label = column  
        fig.add_trace(go.Scatter(x=average_electricity_comp_by_year.index, y=average_electricity_comp_by_year[column],
                                mode='lines+markers',
                                name=legend_label))  
    fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Average Electricity Generation (TWh)',
                    hovermode='x unified',  
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom",
                        y=1.02,  
                        xanchor="right",
                        x=1  
                    ),
                    width=680,
                    height=370  
                    )
    return fig

def region_consumptions(): 
    def country_to_continent(country_name):
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name

    df['Continent'] = df['Entity'].apply(country_to_continent)
    columns = ['Year', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']

    df_energy_all = pd.DataFrame(columns=columns)

    def filter_and_calculate_mean(df, year, column_name):
        row_data = {'Year': year}
        for continent in columns[1:]:
            filtered_df = df[(df['Year'] == year) & (df['Continent'] == continent)]
            mean_value = filtered_df[column_name].mean()
            row_data[continent] = mean_value
        return row_data

    column_name = 'Renewable energy share in the total final energy consumption (%)'
    data_to_concat = []

    for year in range(2000, 2021):
        row_data = filter_and_calculate_mean(df, year, column_name)
        data_to_concat.append(row_data)

    df_energy_all = pd.concat([df_energy_all, pd.DataFrame(data_to_concat)], ignore_index=True)

    df_energy_all['Year'] = df_energy_all['Year'].astype('int64')
    df_energy_cons_melted = pd.melt(df_energy_all, id_vars='Year', var_name='Region', value_name='Energy Consumption')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style='whitegrid')
    palette = sns.color_palette("husl", len(df_energy_cons_melted['Region'].unique()))

    plot = sns.lineplot(ax=ax, data=df_energy_cons_melted, x='Year', y='Energy Consumption', hue='Region', palette=palette)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Renewable energy consumption (%)', fontweight='bold')
    ax.set_title('Renewable Energy Consumption Share (%) by Region', fontweight='bold')
    ax.legend(title='Region')
    ax.margins(x=0)
    years = range(2000, 2021)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45)

    st.pyplot(fig)

    with st.expander("See explanation"):
        st.write("")
        lst = []
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)

def renewable_v_fossil():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Renewable energy share in the total final energy consumption (%)'], df['Electricity from fossil fuels (TWh)'])
    ax.set_xlabel('Renewable energy share in the total final energy consumption (%)')
    ax.set_ylabel('Electricity from fossil fuels (TWh)')
    ax.set_title('Renewable energy share in the total final energy consumption (%) vs Electricity from fossil fuels (TWh)')
    st.pyplot(fig)

    with st.expander("See explanation"):
        st.write("")

def make_donut(selected_year, energy_type, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    df_year = df[df['Year'] == selected_year]
    
    total_fossil_fuels = df_year['Electricity from fossil fuels (TWh)'].sum()
    total_nuclear = df_year['Electricity from nuclear (TWh)'].sum()
    total_renewables = df_year['Electricity from renewables (TWh)'].sum()
    total = total_fossil_fuels + total_nuclear + total_renewables

    percentage_fossil_fuels = (total_fossil_fuels / total) * 100
    percentage_nuclear = (total_nuclear / total) * 100
    percentage_renewables = (total_renewables / total) * 100
    
    source = pd.DataFrame({
        "Topic": ['', energy_type],
        "% value": [100, percentage_renewables if energy_type == 'Renewables' else percentage_nuclear if energy_type == 'Nuclear' else percentage_fossil_fuels]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', energy_type],
        "% value": [100, 0]
    })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45).encode(
        theta="% value",
        color=alt.Color("Topic:N", scale=alt.Scale(domain=[energy_type, ''], range=chart_color), legend=None)
    ).properties(width=160, height=160)
    
    text = plot.mark_text(align='center', color=chart_color[0], fontStyle="italic", fontSize=24, fontWeight=700).encode(text=alt.value(f'{percentage_renewables if energy_type == "Renewables" else percentage_nuclear if energy_type == "Nuclear" else percentage_fossil_fuels:.2f} %'))
    
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45).encode(
        theta="% value",
        color=alt.Color("Topic:N", scale=alt.Scale(domain=[energy_type, ''], range=chart_color), legend=None)
    ).properties(width=160, height=160)
    
    return plot_bg + plot + text

def top10():
    average_energy_consumption_by_country = df.groupby('Entity')['Electricity from renewables (TWh)'].mean()
    top_10_countries = average_energy_consumption_by_country.nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top_10_countries.index, y=top_10_countries.values, ax=ax)
    ax.set_xlabel('Country')
    ax.set_ylabel('Electricity from renewables (TWh)')
    ax.set_title('Top 10 Countries with Highest Average Renewable Sourced Electricity (TWh)')
    ax.set_xticklabels(top_10_countries.index, rotation=45, ha='center') 
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander("See Explanation"):
            st.write("")

def map():
    def plot_map(df, column, title):
        fig = px.choropleth(
            df,
            locations='Entity',
            locationmode='country names',
            color=column,
            hover_name='Entity',
            color_continuous_scale='RdYlGn',
            animation_frame='Year',
            range_color=[0, 100])

        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="white",
            showcountries=True,
            showocean=True,
            oceancolor="LightBlue")

        fig.update_layout(
            title_text=title,
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular',
                showland=True,
                landcolor="white",
                showcountries=True,
                showocean=True,
                oceancolor="LightBlue"),
            width=680,
            height=560,
            dragmode=False,
            uirevision='locked',
            coloraxis_colorbar=dict(
                title=column,
                title_font_size=14,
                title_side='right',
                lenmode='pixels',
                len=300,
                thicknessmode='pixels',
                thickness=15))

        slider_steps = []

        for year in df['Year'].unique():
            step = {
                "args": [
                    [year],
                    {"frame": {"duration": 300, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": str(year),
                "method": "animate"}
            slider_steps.append(step)

        fig.layout.updatemenus[0].buttons[0].args[1]['steps'] = slider_steps

        return fig

    df_subset = df[['Entity', 'Year', 'Access to electricity (% of population)']]

    fig = plot_map(df_subset, 'Access to electricity (% of population)', 'Access to Electricity (% of Population) Over Years')

    st.plotly_chart(fig)
    with st.expander("See Explanation"):
        st.write("Jika dilihat dari skala warna pada peta dunia tersebut adalah:")
        lst = ['Warna merah menunjukkan nilai terendah suatu kota atau wilayah dengan pressentasi penggunaan ataupun akases listrik penduduk terkecil (persentase terkecil penduduk dengan akses listrik).','Warna kuning menunjukkan nilai menengah terhadap suatu kota ataupun wilayah dengan presentase penggunaan ataupun akses listrik yang tergolong menengah dan juga mungkin saja terbatas (tingkat akses listrik moderat).','Warna hijau menunjukkan nilai tertinggi dalam penggunaan ataupun akses listrik terbesar dalam suatu kota ataupun wilayah.  (persentase terbesar penduduk dengan akses listrik).']
        s = ''
        for i in lst:
            s += "- " + i + "\n"
        st.markdown(s)


def predict():

    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Input fields for each column
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input('Year')
        access_to_electricity = st.number_input('Access to electricity (% of population)')
        access_to_clean_fuels = st.number_input('Access to clean fuels for cooking (% of population)')
        renewable_electricity_capacity_per_capita = st.number_input('Renewable electricity Generating Capacity per capita')
        financial_flows_to_developing_countries = st.number_input('Financial flows to developing countries (US $)')
        renewable_energy_share = st.number_input('Renewable energy share in the total final energy consumption (%)')
        electricity_from_fossil_fuels = st.number_input('Electricity from fossil fuels (TWh)')
        electricity_from_nuclear = st.number_input('Electricity from nuclear (TWh)')
    with col2:
        electricity_from_renewables = st.number_input('Electricity from renewables (TWh)')
        low_carbon_electricity = st.number_input('Low-carbon electricity (% electricity)')
        primary_energy_consumption_per_capita = st.number_input('Primary energy consumption per capita (kWh/person)')
        energy_intensity = st.number_input('Energy intensity level of primary energy (MJ/$2017 PPP GDP)')
        co2_emissions = st.number_input('CO2 emissions value by country (kT)')
        renewables_percent = st.number_input('Renewables (% equivalent primary energy)')
        gdp_growth = st.number_input('GDP growth')
        gdp_per_capita = st.number_input('GDP per capita')

    if st.button('Predict'):
        fd = pd.DataFrame({
            'Year': [year],
            'Access to electricity (% of population)': [access_to_electricity],
            'Access to clean fuels for cooking (% of population)': [access_to_clean_fuels],
            'Renewable electricity Generating Capacity per capita': [renewable_electricity_capacity_per_capita],
            'Financial flows to developing countries (US $)': [financial_flows_to_developing_countries],
            'Renewable energy share in the total final energy consumption (%)': [renewable_energy_share],
            'Electricity from fossil fuels (TWh)': [electricity_from_fossil_fuels],
            'Electricity from nuclear (TWh)': [electricity_from_nuclear],
            'Electricity from renewables (TWh)': [electricity_from_renewables],
            'Low-carbon electricity (% electricity)': [low_carbon_electricity],
            'Primary energy consumption per capita (kWh/person)': [primary_energy_consumption_per_capita],
            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': [energy_intensity],
            'CO2 emissions value by country (kT)': [co2_emissions],
            'Renewables (% equivalent primary energy)': [renewables_percent],
            'GDP growth': [gdp_growth],
            'GDP per capita': [gdp_per_capita]
        })

        # Make predictions
        prediction = model.predict(dt)

        # Display prediction result
        st.write('Predicted Renewable consumption Share:', prediction[0],'%')

