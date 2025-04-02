
## Desfecho Clinico
Este projeto é um notebook Jupyter que realiza uma análise preditiva para determinar o risco de pacientes desenvolverem doença cardiovascular em 10 anos (variável alvo TenYearCHD), utilizando dados clínicos do dataset framingham.csv. O objetivo principal é construir um modelo de classificação binária para prever a ocorrência de eventos cardíacos futuros com base em características como idade, sexo, colesterol, pressão arterial, hábitos de fumo e diabetes.
## Principais Etapas do Código:
- **Importação de Bibliotecas**: São importadas bibliotecas para manipulação de dados (numpy, pandas), visualização (matplotlib, seaborn), pré-processamento (StandardScaler, LabelEncoder), balanceamento de classes (SMOTE, SMOTEENN), modelos de classificação (Regressão Logística, Random Forest, XGBoost, entre outros) e métricas de avaliação (acurácia, precisão, AUC-ROC).

- **Análise Exploratória**: 
- Carregamento do dataset com 4240 
- registros e 16 variáveis clínicas.
- Identificação de dados faltantes (ex: glucose tem 9.15% de valores ausentes).
- Estatísticas descritivas das variáveis.
- Verificação da distribuição da variável alvo TenYearCHD.

- **Pré-Processamento**: 
- Configuração para exibir todas as colunas e linhas do DataFrame.
- Tratamento de dados faltantes e normalização.

- **Preparação para Modelagem**:
- Importação de técnicas para lidar com desbalanceamento (SMOTE).
- Definição de validação cruzada estratificada (StratifiedKFold).
- Lista de modelos de classificação a serem testados (Logistic Regression, Random Forest, Gradient Boosting, etc.).

- **Treinando o Modelo**:
Treinando o modelo com base em diferentes algoritimos como Gradient Boosting, Random Forest e etc.
## Ferramanetas utilizadas

- **NumPy**: Manipulação de dados numéricos.

- **Pandas**: Análise e manipulação de dados tabulares.

- **Matplotlib/Seaborn**: Visualização de dados.

 **Scikit-learn**:
- Pré-processamento: StandardScaler, LabelEncoder.
- Balanceamento de classes: SMOTE, SMOTEENN.
- Validação/modelagem: StratifiedKFold, train_test_split, GridSearchCV.
- Métricas: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report.
- Modelos: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier.
- Imbalanced-learn: Técnicas para dados desbalanceados (RandomUnderSampler, SMOTE, SMOTEENN, Pipeline as ImbPipeline).

- **XGBoost**: Modelo de Gradient Boosting (XGBClassifier).

- **LightGBM**: Modelo de Gradient Boosting otimizado (LGBMClassifier).

- **CatBoost**: Modelo de Gradient Boosting com suporte a variáveis categóricas (CatBoostClassifier).

- **TPOT**: Automação de machine learning (TPOTClassifier).
## Stack utilizad

**Jupyter server**

**Python**


## Autores

- [@Gean Vitor](https://www.github.com/GeanVitorM)

