import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

def load_data():
    """Carrega e une todas as bases de dados (Airlines e Airports)."""
    airlines = pd.read_csv('databases/airlines.csv')
    airports = pd.read_csv('databases/airports.csv')
    flights = pd.read_csv('databases/flights.csv', low_memory=False)

    df = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
    df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
    df = df.merge(airports[['IATA_CODE', 'AIRPORT']], left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
    df = df.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRPORT': 'ORIGIN_AIRPORT_NAME'})

    return df


def clean_data(df):
    df_clean = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    cols_to_fix = ['ARRIVAL_DELAY', 'DEPARTURE_DELAY']
    for col in cols_to_fix:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())


    def format_time(time_val):
        if pd.isnull(time_val) or time_val == '': return 0
        try:
            time_str = str(int(float(time_val))).zfill(4)
            hours = int(time_str[:2])
            minutes = int(time_str[2:])
            return hours * 60 + minutes
        except:
            return 0

    df_clean['SCHEDULED_DEPARTURE_MIN'] = df_clean['SCHEDULED_DEPARTURE'].apply(format_time)
    return df_clean


def exploratory_analysis(df):
    plt.figure()

    avg_delay_airport = df.groupby('ORIGIN_AIRPORT_NAME')['DEPARTURE_DELAY'].mean().sort_values(ascending=False).head(
        10)

    sns.barplot(x=avg_delay_airport.values, y=avg_delay_airport.index, hue=avg_delay_airport.index, palette='magma',
                legend=False)
    plt.title('Top 10 Aeroportos com Maiores Atrasos Médios (Origem)')
    plt.xlabel('Média de Atraso na Partida (Minutos)')
    plt.ylabel('Aeroporto')
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.boxplot(data=df, x='ARRIVAL_DELAY', y='AIRLINE_NAME', showfliers=False, palette='Set3', hue='AIRLINE_NAME',
                legend=False)
    plt.title('Variação de Atrasos por Companhia Aérea')
    plt.xlabel('Atraso na Chegada (Minutos)')
    plt.show()


def prepare_for_ml(df):
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    features = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE_MIN', 'DISTANCE', 'AIRLINE_x']
    target = 'IS_DELAYED'

    return df[features + [target]].dropna()


if __name__ == "__main__":
    data = load_data()
    data_cleaned = clean_data(data)
    exploratory_analysis(data_cleaned)

    final_df = prepare_for_ml(data_cleaned)

    print(f"Dataset pronto para ML: {final_df.shape[0]} linhas.")
    print(final_df.head())
