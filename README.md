# AeroInsights

Pipeline de inteligencia analitica para dados de aviacao. O projeto processa registros de voos nos Estados Unidos e entrega analise exploratoria, modelos de machine learning para previsao de atrasos, segmentacao de aeroportos por clustering e um dashboard interativo.

## Estrutura do projeto

```
aeroinsights/
    app.py                 # Dashboard interativo (Streamlit)
    build_database.py      # ETL que gera o banco SQLite
    models.py              # Modelos SQLAlchemy e configuracao do banco
    requirements.txt
    aeroinsights.db        # Banco gerado pelo ETL
    databases/
        airlines.csv       # Companhias aereas
        airports.csv       # Aeroportos (coordenadas, nome, IATA)
        flights.csv        # Registros de voos
```

## Requisitos

- Python 3.10+
- Dependencias listadas em `requirements.txt`

## Instalacao

```bash
pip install -r requirements.txt
```

## Uso

### 1. Construcao do banco

Processa os CSVs e grava duas tabelas no SQLite (`aeroinsights.db`):

- `airport_data` -- metricas agregadas e cluster de cada aeroporto.
- `flights_sample` -- amostra de ate 100 mil voos para o dashboard.

```bash
python build_database.py
```

### 2. Dashboard

Requer que o banco ja esteja construido (passo anterior).

```bash
streamlit run app.py
```

## Pipeline de dados

1. **Carga** -- merge de `flights.csv`, `airlines.csv` e `airports.csv`.
2. **Limpeza** -- remocao de voos cancelados/desviados; imputacao de nulos pela mediana.
3. **Feature engineering** -- estacao do ano calculada por hemisferio e mes.
4. **Clustering** -- K-Means (k=3) sobre volume de voos e atraso medio por aeroporto.
5. **Persistencia** -- tabelas `airport_data` e `flights_sample` gravadas no SQLite.

## Dashboard

O dashboard Streamlit apresenta:

- Resumo operacional (total de aeroportos, atraso medio global, volume de voos).
- Mapa interativo de atrasos por localizacao.
- Ranking dos 10 aeroportos mais criticos.
- Impacto sazonal nos atrasos.
- Desempenho por companhia aerea.
- Segmentacao estrategica por clusters com grafico e tabela.

## Banco de dados

O acesso ao SQLite e feito via SQLAlchemy. Os modelos estao definidos em `models.py`:

- `AirportData` -- tabela `airport_data`.
- `FlightSample` -- tabela `flights_sample`.

A connection string e configurada na variavel `DATABASE_URL` em `models.py`.

## Tecnologias

| Componente       | Biblioteca              |
|------------------|-------------------------|
| Manipulacao      | pandas                  |
| Visualizacao     | plotly                  |
| Clustering       | scikit-learn            |
| Dashboard        | streamlit               |
| Banco de dados   | SQLite + SQLAlchemy     |

