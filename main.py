import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurações Globais
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
MAX_SAMPLE_SIZE = 100000

# Garantir pasta de resultados
if not os.path.exists('results'):
    os.makedirs('results')


def load_data():
    """Carrega e une todas as bases de dados enriquecendo com geo-localização."""
    print("--- Carregando dados ---")
    airlines = pd.read_csv('databases/airlines.csv')
    airports = pd.read_csv('databases/airports.csv')
    flights = pd.read_csv('databases/flights.csv', low_memory=False)

    # Merge Airlines
    df = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')

    # Merge Airports (trazendo Latitude e Longitude para análise avançada)
    df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
    df = df.merge(airports[['IATA_CODE', 'AIRPORT', 'LATITUDE', 'LONGITUDE']],
                  left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')

    df = df.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRPORT': 'ORIGIN_AIRPORT_NAME'})
    return df


def clean_data(df):
    """Limpeza e formatação de horários."""
    print("--- Limpando dados ---")
    df_clean = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    # Preenchimento de nulos
    cols_to_fix = ['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'LATITUDE', 'LONGITUDE']
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


def feature_engineering_advanced(df):
    """Geração de variáveis de destaque: Sazonalidade Global e Red-Eye."""
    print("--- Executando Engenharia de Features Avançada ---")

    # 1. Sazonalidade baseada no Hemisfério (Latitude)
    def get_season(row):
        m, lat = row['MONTH'], row['LATITUDE']
        is_north = lat >= 0
        if is_north:
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

    # 2. Voo de Madrugada (Red-Eye) - Frequentemente associado a menos atrasos de tráfego
    df['IS_RED_EYE'] = (df['SCHEDULED_DEPARTURE_MIN'] < 300).astype(int)

    # 3. Recuperação de tempo (Anomalia: partida atrasada mas chegada pontual)
    df['TIME_RECOVERY'] = df['DEPARTURE_DELAY'] - df['ARRIVAL_DELAY']

    return df


def exploratory_analysis_advanced(df):
    print("--- Gerando Visualizações de Destaque ---")

    # 1. Mapa Geográfico de Risco de Atraso
    plt.figure(figsize=(12, 7))
    geo_stats = df.groupby(['LONGITUDE', 'LATITUDE'])['DEPARTURE_DELAY'].mean().reset_index()
    sc = plt.scatter(geo_stats['LONGITUDE'], geo_stats['LATITUDE'],
                     c=geo_stats['DEPARTURE_DELAY'], cmap='YlOrRd', alpha=0.6)
    plt.colorbar(sc, label='Atraso Médio (Min)')
    plt.title('AeroInsights: Mapa Geográfico de Risco de Atraso nos EUA')
    plt.savefig('results/mapa_geografico_atrasos.png')
    plt.close()

    # 2. Atrasos por Estação do Ano
    plt.figure()
    sns.barplot(data=df, x='SEASON', y='ARRIVAL_DELAY', palette='coolwarm', hue='SEASON', legend=False)
    plt.title('Impacto da Sazonalidade nos Atrasos de Chegada')
    plt.savefig('results/eda_sazonalidade.png')
    plt.close()

    # 3. Recuperação de Tempo (Histograma)
    plt.figure()
    sns.histplot(df['TIME_RECOVERY'], bins=50, color='teal', kde=True)
    plt.title('Distribuição de Recuperação de Tempo em Voo')
    plt.xlabel('Minutos Recuperados durante o Voo')
    plt.savefig('results/anomalia_recuperacao.png')
    plt.close()


def prepare_for_ml(df):
    """Prepara o dataset incluindo as novas variáveis convertidas."""
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    # Incluímos SEASON para o modelo (precisará de dummies)
    features = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE_MIN', 'DISTANCE', 'AIRLINE_x', 'SEASON', 'IS_RED_EYE']
    target = 'IS_DELAYED'
    return df[features + [target]].dropna()


def train_and_evaluate_models(df):
    print("--- Iniciando Modelagem Supervisionada ---")
    df_sample = df.sample(n=min(MAX_SAMPLE_SIZE, len(df)), random_state=42).copy()

    # Get Dummies para AIRLINE e SEASON
    df_sample = pd.get_dummies(df_sample, columns=['AIRLINE_x', 'SEASON'], drop_first=True)

    X = df_sample.drop('IS_DELAYED', axis=1)
    y = df_sample['IS_DELAYED']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelos com Pesos Balanceados
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1,
                                      random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\n" + "=" * 40)
    print("MÉTRICAS: RANDOM FOREST (AVANÇADO)")
    print(classification_report(y_test, rf_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='RdPu')
    plt.title('Matriz de Confusão - Modelo Final AeroInsights')
    plt.savefig('results/ml_matriz_confusao.png')
    plt.close()

    return rf_model


def perform_clustering(df_full):
    print("--- Iniciando Clusterização de Aeroportos ---")
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
                    s=100)
    plt.title('Segmentação AeroInsights: Perfis de Aeroportos')
    plt.savefig('results/cluster_segmentacao_aeroportos.png')
    plt.close()

    print("\n--- Perfil Médio dos Clusters ---")
    print(airport_stats.groupby('CLUSTER').mean().sort_values(by='AVG_DEPARTURE_DELAY'))
    return airport_stats


if __name__ == "__main__":
    # 1. Pipeline de Dados
    data = load_data()
    data_cleaned = clean_data(data)
    data_enriched = feature_engineering_advanced(data_cleaned)

    # 2. Análises e Visualizações
    exploratory_analysis_advanced(data_enriched)

    # 3. Machine Learning
    final_df = prepare_for_ml(data_enriched)
    train_and_evaluate_models(final_df)
    perform_clustering(data_enriched)

    print("\n--- Projeto AeroInsights finalizado. Verifique a pasta /results ---")