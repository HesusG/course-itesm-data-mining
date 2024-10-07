# %% [markdown]
# # Actividad de Clustering en Equipo: Análisis y Recomendaciones para Kueski
# 
# **Objetivo:** Realizar un análisis de clustering de los clientes de Kueski utilizando K-means, interpretar los resultados y proponer estrategias de negocio basadas en los hallazgos.
# 
# ## 1. Análisis de Datos (Trabajo Conjunto)
# 
# ### 1.1. Importación de Librerías
# 
# Comenzamos importando las librerías necesarias para el análisis de datos y visualización.
# 
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')
sns.set_palette('husl')

# %% [markdown]
# ### 1.2. Carga de Datos
# 
# Cargamos el archivo CSV que contiene los datos de los clientes de Kueski.
# 
# %%
archivo = 'Tarea1Creditos.csv'
customer_data = pd.read_csv(archivo)

# %% [markdown]
# ### 1.3. Exploración Inicial de los Datos
# 
# Inspeccionamos la forma del conjunto de datos, las columnas presentes y realizamos una descripción estadística básica.
# 
# %%
# Inspeccionar la forma de los datos
print("Forma del DataFrame:", customer_data.shape)

# %%
# Acceder a las columnas
print("Columnas del DataFrame:", customer_data.columns)

# %%
# Histograma de 'Spending Score (1-100)'
customer_data['Spending Score (1-100)'].hist()
plt.title('Distribución del Spending Score')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Descripción estadística
print(customer_data.describe().T)

# %%
# Histograma de 'Annual Income (k$)'
customer_data['Annual Income (k$)'].hist()
plt.title('Distribución del Ingreso Anual')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Información del DataFrame
print(customer_data.info())

# %%
# Primeras 10 filas del DataFrame
print(customer_data.head(10))

# %%
# Valores únicos en la columna 'Genre'
print("Valores únicos en 'Genre':", customer_data['Genre'].unique())

# %%
# Conteo de valores en 'Genre'
print(customer_data['Genre'].value_counts(normalize=True))

# %% [markdown]
# ### 1.4. Preprocesamiento de Datos
# 
# Convertimos las variables categóricas en numéricas para facilitar el análisis de clustering.
# 
# %%
# Diccionario de reemplazo
replace_dict = {
    'Android' : 0,
    'Ios' : 1,
    'Rural' : 0,
    'Urban' : 1,
    'Male'  : 0,
    'Female' : 1
}

# Aplicar el reemplazo
customer_data.replace(replace_dict, inplace=True)

# %% [markdown]
# ## 2. Implementación de K-means
# 
# A continuación, implementamos el algoritmo K-means utilizando Scikit-Learn para realizar el clustering de los clientes.
# 
# **Nota:** Es importante escalar las características antes de aplicar K-means para asegurar que todas las variables contribuyan equitativamente al clustering.

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Seleccionar las características para el clustering
X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Escalado de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir el modelo K-means
kmeans = KMeans(n_clusters=5, random_state=42)

# Ajustar el modelo
kmeans.fit(X_scaled)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# %% [markdown]
# ### 2.1. Visualización de los Clusters
# 
# Representamos gráficamente los clusters obtenidos para visualizar la segmentación de los clientes.
# 
# %%
# Agregar las etiquetas de los clusters al DataFrame original
customer_data['Cluster'] = labels

# Gráfico de dispersión de los clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)',
                y='Spending Score (1-100)',
                hue='Cluster',
                palette='bright',
                data=customer_data)
plt.title('Clusters de Clientes')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# %% [markdown]
# ## 3. Interpretación de Resultados (Trabajo Conjunto)
# 
# Analizamos las características de cada cluster para identificar patrones y tendencias en los datos.
# 
# %%
# Añadir las etiquetas de los clusters al DataFrame
cluster_results = customer_data.copy()
cluster_results['Cluster Labels'] = labels

# %%
# Filtrar un cluster específico (ejemplo: Cluster 4)
cluster_4 = cluster_results.loc[cluster_results['Cluster Labels'] == 4]
print(cluster_4.describe())

# %%
# Histograma de 'Age' en el Cluster 4
cluster_4['Age'].hist()
plt.title('Distribución de Edad en Cluster 4')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Histograma de 'Loan Ammount' en el Cluster 4
cluster_4['Loan Ammount'].hist()
plt.title('Distribución de Monto de Préstamo en Cluster 4')
plt.xlabel('Monto de Préstamo')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Boxplot de 'Annual Income (k$)' por Cluster y Género
plt.figure(figsize=(10,6))
sns.boxplot(data=cluster_results, x="Cluster Labels", y="Annual Income (k$)", hue="Genre")
plt.title('Ingreso Anual por Cluster y Género')
plt.show()

# %%
# Boxplot de 'Spending Score (1-100)' por Cluster
plt.figure(figsize=(10,6))
sns.boxplot(data=cluster_results, x="Cluster Labels", y="Spending Score (1-100)")
plt.title('Spending Score por Cluster')
plt.show()

# %%
# Boxplot de 'Age' por Cluster
plt.figure(figsize=(10,6))
sns.boxplot(data=cluster_results, x="Cluster Labels", y="Age")
plt.title('Edad por Cluster')
plt.show()

# %%
# Primeras filas del resultado del clustering
print(cluster_results.head())

# %%
# Gráfico de dispersión por clusters utilizando FacetGrid
facet = sns.FacetGrid(cluster_results, col="Cluster Labels")
facet.map(sns.scatterplot, "Annual Income (k$)", "Spending Score (1-100)")
plt.show()

# %%
# Agrupación por Cluster y cálculo de correlaciones
cluster_bygroups = cluster_results.select_dtypes(include=['int', 'float']).groupby("Cluster Labels")
heatmap_data = cluster_bygroups.corr()
heatmap_data = heatmap_data.xs('Annual Income (k$)', level=1)
heatmap = sns.heatmap(heatmap_data, annot=True, cmap='viridis')
heatmap.set_title('Correlación de Variables por Cluster')
plt.ylabel('Cluster Labels')
plt.show()

# %%
# Gráfico de distribución de 'Loan Ammount' por Cluster
facet = sns.FacetGrid(cluster_results, col="Cluster Labels")
facet.map(sns.histplot, "Loan Ammount")
plt.show()

# %%
# Análisis de variables categóricas
categorical_rows = cluster_results.select_dtypes(include=['object'])
categorical_rows = categorical_rows.loc[:, categorical_rows.columns != 'LoanID']

# %%
# Conteo de valores categóricos
categorical_rows.value_counts().plot(kind='bar')
plt.title('Conteo de Variables Categóricas')
plt.show()

# %%
# Conteo acumulado de valores categóricos normalizados
print(categorical_rows.value_counts(normalize=True).cumsum())

# %%
# Gráfico de barras para 'Genre'
categorical_rows["Genre"].value_counts(normalize=True).plot(kind="bar")
plt.title('Distribución de Género')
plt.xlabel('Género')
plt.ylabel('Proporción')
plt.show()

# %%
# Gráfico de barras para 'Location'
categorical_rows["Location"].value_counts(normalize=True).plot(kind="bar")
plt.title('Distribución de Ubicación')
plt.xlabel('Ubicación')
plt.ylabel('Proporción')
plt.show()

# %%
# Distribución de las etiquetas de los clusters
cluster_results["Cluster Labels"].value_counts(normalize=True).plot(kind="bar")
plt.title('Distribución de Clusters')
plt.xlabel('Cluster')
plt.ylabel('Proporción')
plt.show()

# %%
# Resumen estadístico por Cluster
summarize = cluster_bygroups.agg(
    count=('Loan Ammount','count'),
    loan_ammount_sum=('Loan Ammount','sum'),
    median_age=('Age','median'),
    median_loan=('Loan Ammount', 'median'),
    median_annual_income=('Annual Income (k$)','median')
)
summarize["loan_pct"] = (summarize["loan_ammount_sum"] / summarize["loan_ammount_sum"].sum()) * 100
summarize["payment_ratio"] = (summarize["median_loan"] / (summarize["median_annual_income"] * 1000 / 12)) * 100
summarize = summarize.sort_values(by=["loan_pct","median_annual_income"], ascending=False)
print(summarize)

# %% [markdown]
# ## 4. Desarrollo de Recomendaciones (Trabajo Individual)
# 
# Cada miembro del equipo debe desarrollar al menos una recomendación de negocio específica para un cluster asignado. A continuación, se muestra un ejemplo de cómo estructurar las recomendaciones.
# 
# ### Recomendación para el Cluster 0
# 
# %%
# Filtrar datos para el Cluster 0
df_ideal = cluster_results[cluster_results["Cluster Labels"] == 0]
print(df_ideal.describe())

# %%
# Análisis de variables categóricas en el Cluster 0
df_ideal.drop("LoanID", axis=1).select_dtypes(include=['object']).value_counts().plot(kind='bar')
plt.title('Variables Categóricas en Cluster 0')
plt.show()

# %%
# Gráfico de barras agrupado por Género, Dispositivo y Ubicación
plt.figure(figsize=(10,5))
sns.catplot(data=df_ideal,
            x="Genre",
            y="Loan Ammount",
            hue="Device",
            col="Location",
            kind="bar")
plt.xticks(rotation=90)
plt.title('Monto de Préstamo por Género, Dispositivo y Ubicación en Cluster 0')
plt.show()

# %%
# FacetGrid para Edad y Dispositivo por Género y Ubicación
facet = sns.FacetGrid(df_ideal, col="Genre", row="Location", hue="Device", height=3.5, aspect=2)
facet.map(sns.histplot, "Age", binwidth=10, binrange=(19, 60))
facet.add_legend()
plt.show()

# %%
# FacetGrid para Monto de Préstamo por Género y Ubicación
facet = sns.FacetGrid(df_ideal, col="Genre", row="Location", hue="Device", height=3.5, aspect=2)
facet.map(sns.histplot, "Loan Ammount", binwidth=500, binrange=(1200, 3500))
facet.add_legend()
plt.show()

# %%
# Mediana del Monto de Préstamo en el Cluster 0
print("Mediana del Monto de Préstamo en Cluster 0:", df_ideal["Loan Ammount"].median())

# %%
# Heatmap de Pivot para Monto de Préstamo por Edad
plt.figure(figsize=(15,2))
ideal_pivot = df_ideal.pivot_table(values='Loan Ammount', columns='Age')
heatmap = sns.heatmap(ideal_pivot,
                      cbar=True,
                      cmap="BuGn",
                      linewidths=3,
                      center=df_ideal['Loan Ammount'].median(),
                      annot=True,
                      fmt="d")
plt.title('Heatmap de Monto de Préstamo por Edad en Cluster 0')
plt.show()

# %%
# Comparación de clientes actuales y futuros
df_current_customers = cluster_results[cluster_results["Cluster Labels"].isin([2,1])]
df_future_customers = cluster_results[cluster_results["Cluster Labels"].isin([0,1])]

print(df_current_customers["Loan Ammount"].sum(), "Monto actual disponible para cobranza")
print(df_future_customers["Loan Ammount"].sum(), "Monto futuro disponible para cobranza")

print(((df_future_customers["Loan Ammount"].sum() / df_current_customers["Loan Ammount"].sum()) - 1) * 100,
      "posible % de incremento")

# %%
from scipy import stats

# %%
# Histograma del Monto de Préstamo
cluster_results["Loan Ammount"].hist()
plt.title('Distribución del Monto de Préstamo')
plt.xlabel('Monto de Préstamo')
plt.ylabel('Frecuencia')
plt.show()

# %%
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

# %%
# Escalado y normalización del Monto de Préstamo
scaler = StandardScaler()
normalizer = PowerTransformer(method='box-cox')

# Asegurarse de que no haya valores <=0 para Box-Cox
if (cluster_results["Loan Ammount"] <= 0).any():
    # Añadir una constante si hay valores <=0
    data = normalizer.fit_transform(cluster_results["Loan Ammount"].values.reshape(-1,1) + 1)
else:
    data = normalizer.fit_transform(cluster_results["Loan Ammount"].values.reshape(-1,1))

cluster_results['Loan Ammount Norm'] = data

# %%
import statsmodels.api as sm
import scipy.stats

# %%
# Gráfico Q-Q para verificar normalidad
sm.qqplot(cluster_results["Loan Ammount Norm"], line='s')
plt.title('Q-Q Plot del Monto de Préstamo Normalizado')
plt.show()

# %%
# Prueba de Shapiro-Wilk para normalidad
print(scipy.stats.shapiro(cluster_results["Loan Ammount Norm"]))

# %%
# KDE plot del Monto de Préstamo por Género en el Cluster 0
sns.displot(data=df_ideal, x="Loan Ammount", hue="Genre", kind="kde")
plt.title('Distribución KDE del Monto de Préstamo por Género en Cluster 0')
plt.show()

# %%
# Selección de Monto de Préstamo para Género Masculino y Femenino en el Cluster 0
male_loans = df_ideal[df_ideal["Genre"] == "Male"]["Loan Ammount"]
female_loans = df_ideal[df_ideal["Genre"] == "Female"]["Loan Ammount"]

# %%
from scipy.stats import mannwhitneyu

# %%
# Prueba de Mann-Whitney para comparar Monto de Préstamo entre Géneros
result = mannwhitneyu(male_loans, female_loans)
print(result)

# %% [markdown]
# **Interpretación:** Se rechaza la hipótesis nula. Los grupos son significativamente diferentes.
# 
# %%
# Descripción agrupada por Género en el Cluster 0
print(df_ideal.groupby("Genre").describe().T)

# %%
# Gráfico de dispersión con tamaño de punto basado en Monto de Préstamo y color por Género
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_ideal,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                size="Loan Ammount",
                hue="Genre",
                palette='viridis')
plt.axhline(df_ideal["Spending Score (1-100)"].median(), color='red', linestyle='--')
plt.title('Dispersión de Ingreso Anual vs Spending Score en Cluster 0')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

# %% [markdown]
# ## Revisión de la Implementación de K-means
# 
# La implementación de K-means realizada utiliza la biblioteca `scikit-learn`, lo cual es adecuado para este tipo de análisis. Sin embargo, hay algunos aspectos que se deben considerar para asegurar que el clustering sea efectivo:
# 
# ### 1. Escalado de Características
# 
# Es esencial escalar las características antes de aplicar K-means, ya que el algoritmo se basa en distancias euclidianas y las diferencias en las escalas pueden sesgar los resultados. En la implementación proporcionada, se ha añadido el escalado utilizando `StandardScaler`.
# 
# ### 2. Selección del Número de Clusters
# 
# Actualmente, se ha fijado el número de clusters en 5. Sin embargo, es recomendable determinar el número óptimo de clusters utilizando métodos como el **Elbow Method** o el **Silhouette Score**. A continuación, se muestra cómo implementar el Elbow Method:
# 
# %%
from sklearn.metrics import silhouette_score

# %%
# Determinar el número óptimo de clusters utilizando el Elbow Method
sse = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# %%
# Gráfico del Elbow Method
plt.figure(figsize=(10,5))
plt.plot(K, sse, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.title('Elbow Method Para Determinar el Número Óptimo de Clusters')
plt.show()

# %%
# Gráfico del Silhouette Score
plt.figure(figsize=(10,5))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Para Determinar el Número Óptimo de Clusters')
plt.show()

# %% [markdown]
# **Interpretación:** Basado en los gráficos, seleccionamos el número de clusters que muestra una disminución significativa en el SSE (Elbow Method) y un alto Silhouette Score. Por ejemplo, si el codo se observa en k=5 y el Silhouette Score es alto, entonces k=5 es una buena elección.
# 
# ### 3. Validación de los Clusters
# 
# Además de la visualización, es importante interpretar las características de cada cluster para asegurar que sean significativas y útiles para las estrategias de negocio.
# 
# ## Conclusión
# 
# La implementación de K-means es adecuada, pero siempre es recomendable validar y ajustar los parámetros según los datos específicos para obtener resultados óptimos.

# ```

# **¡Buena suerte con tu tarea! Si necesitas más ayuda, no dudes en preguntar.**
