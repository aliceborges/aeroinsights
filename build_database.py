import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models import engine, init_db


def build_backend():
    """
    Build the AeroInsights database backend with ETL pipeline.

    This function performs the following operations:
    1. Loads airline, airport, and flight data from CSV files
    2. Merges datasets and standardizes column names
    3. Cleans data by removing cancelled/diverted flights and handling nulls
    4. Engineers season features based on geographic latitude
    5. Performs K-Means clustering on airport statistics
    6. Saves processed data to SQLite database

    The function creates two tables:
    - airport_data: Aggregated metrics and clusters for each airport
    - flights_sample: Sample of flight records for dashboard visualization
    """
    airlines = pd.read_csv('databases/airlines.csv')
    airports = pd.read_csv('databases/airports.csv')
    flights = pd.read_csv('databases/flights.csv', low_memory=False)
    flights.columns = [c.upper() for c in flights.columns]
    airlines.columns = [c.upper() for c in airlines.columns]
    airports.columns = [c.upper() for c in airports.columns]

    df = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
    df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
    df = df.merge(airports[['IATA_CODE', 'AIRPORT', 'LATITUDE', 'LONGITUDE']],
                  left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')

    df = df.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRPORT': 'ORIGIN_AIRPORT_NAME'})

    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    for col in ['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'LATITUDE', 'LONGITUDE']:
        df[col] = df[col].fillna(df[col].median())

    def get_season(row):
        m, lat = row['MONTH'], row['LATITUDE']
        if lat >= 0:
            if m in [12, 1, 2]:
                return 'Winter'
            elif m in [3, 4, 5]:
                return 'Spring'
            elif m in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        else:
            if m in [12, 1, 2]:
                return 'Summer'
            elif m in [3, 4, 5]:
                return 'Autumn'
            elif m in [6, 7, 8]:
                return 'Winter'
            else:
                return 'Spring'

    df['SEASON'] = df.apply(get_season, axis=1)

    airport_stats = df.groupby(['ORIGIN_AIRPORT_NAME', 'LATITUDE', 'LONGITUDE']).agg({
        'DEPARTURE_DELAY': 'mean',
        'MONTH': 'count'
    }).reset_index().rename(columns={'MONTH': 'TOTAL_FLIGHTS', 'DEPARTURE_DELAY': 'AVG_DELAY'})

    scaler = StandardScaler()
    scaled = scaler.fit_transform(airport_stats[['AVG_DELAY', 'TOTAL_FLIGHTS']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    airport_stats['CLUSTER'] = kmeans.fit_predict(scaled)

    init_db()

    airport_stats.to_sql('airport_data', engine, if_exists='replace', index=False)

    df[['AIRLINE_NAME', 'ARRIVAL_DELAY', 'SEASON', 'MONTH']].sample(n=min(100000, len(df))).to_sql(
        'flights_sample',
        engine,
        if_exists='replace',
        index=False
    )


if __name__ == "__main__":
    build_backend()
