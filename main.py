import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
MAX_SAMPLE_SIZE=100000


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
        except (ValueError, TypeError):
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
    plt.tight_layout()
    plt.savefig('results/eda_top_aeroportos.png')
    plt.close()

    plt.figure()
    sns.boxplot(data=df, x='ARRIVAL_DELAY', y='AIRLINE_NAME', showfliers=False, palette='Set3', hue='AIRLINE_NAME',
                legend=False)
    plt.title('Variação de Atrasos por Companhia Aérea')
    plt.savefig('results/eda_boxplot_airlines.png')
    plt.close()
    print("Gráficos da EDA salvos em /results")


def prepare_for_ml(df):
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    features = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE_MIN', 'DISTANCE', 'AIRLINE_x']
    target = 'IS_DELAYED'

    return df[features + [target]].dropna()


def train_and_evaluate_models(df):
    df_sample = df.sample(n=min(MAX_SAMPLE_SIZE, len(df)), random_state=42).copy()
    df_sample = pd.get_dummies(df_sample, columns=['AIRLINE_x'], drop_first=True)

    X = df_sample.drop('IS_DELAYED', axis=1)
    y = df_sample['IS_DELAYED']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1,
                                      random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\n" + "=" * 40)
    print("MÉTRICAS: REGRESSÃO LOGÍSTICA (BALANCED)")
    print(classification_report(y_test, lr_preds))

    print("\n" + "=" * 40)
    print("MÉTRICAS: RANDOM FOREST (BALANCED)")
    print(classification_report(y_test, rf_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='RdPu')
    plt.title('Matriz de Confusão Equilibrada - Random Forest')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig('results/ml_matriz_confusao.png')
    plt.close()

    return rf_model


def perform_clustering(df_full):
    airport_stats = df_full.groupby('ORIGIN_AIRPORT_NAME').agg({
        'DEPARTURE_DELAY': 'mean',
        'MONTH': 'count'
    }).rename(columns={'MONTH': 'TOTAL_FLIGHTS', 'DEPARTURE_DELAY': 'AVG_DEPARTURE_DELAY'})

    scaler = StandardScaler()
    stats_scaled = scaler.fit_transform(airport_stats)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    airport_stats['CLUSTER'] = kmeans.fit_predict(stats_scaled)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=airport_stats, x='TOTAL_FLIGHTS', y='AVG_DEPARTURE_DELAY', hue='CLUSTER', palette='viridis',
                    s=100, alpha=0.7)
    plt.title('AeroInsights: Segmentação de Aeroportos')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('results/cluster_segmentacao_aeroportos.png')  # Exportação
    plt.close()
    print("Gráfico de Clusterização salvo em /results")

    return airport_stats


if __name__ == "__main__":
    data = load_data()
    data_cleaned = clean_data(data)
    exploratory_analysis(data_cleaned)

    final_df = prepare_for_ml(data_cleaned)
    train_and_evaluate_models(final_df)
    airport_clusters = perform_clustering(data_cleaned)

    print(f"Dataset pronto para ML: {final_df.shape[0]} linhas.")
    print(final_df.head())
