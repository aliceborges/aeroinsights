import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import select

from models import engine, AirportData, FlightSample

st.set_page_config(page_title="AeroInsights Dashboard", layout="wide")

st.title("AeroInsights: Aviation Intelligence")
st.markdown("Interactive analysis of flight delays and airport segmentation.")


@st.cache_data
def load_data():
    """
    Load airport and flight sample data from the database using SQLAlchemy.

    Returns:
        tuple: DataFrames containing airports and flight samples
    """
    with engine.connect() as conn:
        airports = pd.read_sql(select(AirportData), conn)
        samples = pd.read_sql(
            select(
                FlightSample.AIRLINE_NAME,
                FlightSample.ARRIVAL_DELAY,
                FlightSample.SEASON,
                FlightSample.MONTH,
            ),
            conn,
        )
    return airports, samples


try:
    airports_df, samples_df = load_data()

    st.subheader("Operational Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Airports Analyzed", len(airports_df))
    c2.metric("Global Average Delay", f"{airports_df['AVG_DELAY'].mean():.2f} min")
    c3.metric("Flight Volume", f"{airports_df['TOTAL_FLIGHTS'].sum():,}".replace(",", "."))

    st.divider()

    map_col, ranking_col = st.columns([2, 1])
    with map_col:
        st.markdown("Hover over points to see airport names. Redder colors indicate higher delays.")

        map_data = airports_df.copy()

        fig = px.scatter_mapbox(
            map_data,
            lat="LATITUDE",
            lon="LONGITUDE",
            hover_name="ORIGIN_AIRPORT_NAME",
            hover_data={
                "AVG_DELAY": ':.2f',
                "TOTAL_FLIGHTS": True,
                "LATITUDE": False,
                "LONGITUDE": False
            },
            color="AVG_DELAY",
            size="TOTAL_FLIGHTS",
            color_continuous_scale="Reds",
            zoom=3,
            mapbox_style="carto-positron"
        )

        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    st.divider()

    with ranking_col:
        st.subheader("Top 10 Critical Airports")
        top_10 = airports_df.sort_values('AVG_DELAY', ascending=False).head(10).copy()
        top_10 = top_10.rename(columns={'ORIGIN_AIRPORT_NAME': 'Airport', 'AVG_DELAY': 'Minutes'})
        st.bar_chart(top_10, x='Airport', y='Minutes', color='#FF4B4B')

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Seasonality Impact")
        season_data = samples_df.groupby('SEASON')['ARRIVAL_DELAY'].mean().reset_index()
        season_data = season_data.rename(columns={'SEASON': 'Season', 'ARRIVAL_DELAY': 'Average Delay'})
        st.bar_chart(season_data, x='Season', y='Average Delay', color='#29B5E8')

    with col_b:
        st.subheader("Performance by Airline")
        airline_data = samples_df.groupby('AIRLINE_NAME')['ARRIVAL_DELAY'].mean().sort_values(
            ascending=False).reset_index()
        airline_data = airline_data.rename(columns={'AIRLINE_NAME': 'Airline', 'ARRIVAL_DELAY': 'Average Delay'})
        st.bar_chart(airline_data, x='Airline', y='Average Delay', color='#155F7A')

    st.divider()
    st.subheader("Strategic Segmentation (K-Means)")

    tab_grafico, tab_lista = st.tabs(["Cluster Chart", "Airport List"])

    with tab_grafico:
        scatter_data = airports_df.rename(columns={
            'TOTAL_FLIGHTS': 'Flight Volume',
            'AVG_DELAY': 'Average Delay (min)',
            'CLUSTER': 'Cluster'
        })
        st.scatter_chart(scatter_data, x='Flight Volume', y='Average Delay (min)', color='Cluster',
                         size='Flight Volume')

    with tab_lista:
        st.dataframe(
            airports_df[['CLUSTER', 'ORIGIN_AIRPORT_NAME', 'TOTAL_FLIGHTS', 'AVG_DELAY']]
            .sort_values(by=['CLUSTER', 'AVG_DELAY'], ascending=[True, False]),
            column_config={
                "CLUSTER": "Cluster",
                "ORIGIN_AIRPORT_NAME": "Airport Name",
                "TOTAL_FLIGHTS": "Total Flights",
                "AVG_DELAY": st.column_config.NumberColumn("Average Delay (min)", format="%.2f")
            }, hide_index=True, use_container_width=True
        )

    st.info("Cluster Legend: Cluster 0: Small/Unstable | Cluster 1: Efficient | Cluster 2: Critical Hubs")

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Please check if 'aeroinsights.db' file exists in the directory.")
