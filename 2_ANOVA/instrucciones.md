### 1. **Carga y Preprocesamiento de Datos**
1. **Importa las librerías**: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `statsmodels.formula.api`, `scipy.stats`.
2. **Carga el CSV** y verifica las primeras filas con `df.head()`.
3. **Elimina valores nulos** de las columnas 'Broad Bean Origin' y 'Rating' con `df.dropna()`.
4. **Convierte** 'Cocoa Percent' a tipo numérico con `astype(float)`.
5. **Filtra** los países de interés en 'Broad Bean Origin'  por la alta cardinalidad de origenes solo usar
lo siguientes:
selected_origins = ['Mexico', 'Colombia', 'Ghana', 'Nigeria', 'Cameroon', 'Madagascar', 'Togo', 'Uganda', 'Sierra Leone']


### 2. **Análisis Exploratorio de Datos (EDA)**
6. **Visualiza las distribuciones** de 'Rating' por origen con `sns.boxplot()` y `sns.violinplot()`.

### 3. **Pruebas de Supuestos**
7. **Prueba la normalidad** por grupo usando `shapiro()`.
8. **Realiza el test de Levene** para homogeneidad de varianzas con `levene()`.

### 4. **Análisis ANOVA**
9. **Ajusta un modelo OLS** con `ols('Rating ~ C(Broad_Bean_Origin)')` y realiza ANOVA con `anova_lm()`.
10. **Interpreta el valor p** de ANOVA para verificar diferencias significativas.


### 5. **Prueba Post-hoc y Conclusiones**
11. **Si ANOVA es significativo**, realiza la prueba post-hoc con `pairwise_tukeyhsd()`.
12. **Visualiza los resultados** del test de Tukey usando `tukey.plot_simultaneous()`.
13. **Concluye** si existen diferencias significativas entre los grupos.