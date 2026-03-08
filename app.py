import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import select

from models import engine, AirportData, FlightSample, ModelMetrics, FeatureImportance, ConfusionMatrix

st.set_page_config(page_title="AeroInsights Dashboard", layout="wide")

st.title("AeroInsights: Aviation Intelligence")
st.markdown("Interactive analysis of flight delays and airport segmentation.")


@st.cache_data
def load_data():
    """
    Load airport and flight sample data from the database using SQLAlchemy.

    Returns:
        tuple: DataFrames containing airports, flight samples, and ML metrics
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
        metrics = pd.read_sql(select(ModelMetrics), conn)
        feat_imp = pd.read_sql(select(FeatureImportance), conn)
        cm = pd.read_sql(select(ConfusionMatrix), conn)

    return airports, samples, metrics, feat_imp, cm


try:
    airports_df, samples_df, metrics_df, feat_imp_df, cm_df = load_data()

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

    st.divider()
    st.subheader("Supervised Learning: Delay Prediction (Random Forest)")
    st.markdown("Predicting if a flight will be delayed (>15 min) based on season, airline, and distance.")

    m_dict = dict(zip(metrics_df['Metric'], metrics_df['Value']))

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Accuracy", f"{m_dict.get('Accuracy', 0):.1%}")
    col_m2.metric("Precision (Delayed)", f"{m_dict.get('Precision', 0):.1%}")
    col_m3.metric("Recall (Delayed)", f"{m_dict.get('Recall', 0):.1%}")

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm_values = cm_df[['Predicted_OnTime', 'Predicted_Delayed']].values
        fig_cm = px.imshow(
            cm_values, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['On Time', 'Delayed'], y=['On Time', 'Delayed']
        )
        fig_cm.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown("**Top Feature Importances**")
        feat_chart_data = feat_imp_df.rename(columns={'Importance': 'Weight'})
        st.bar_chart(feat_chart_data, x='Feature', y='Weight', color='#8B5CF6', height=350)

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Please check if 'aeroinsights.db' file exists in the directory.")