# 🍄 Clasificación de Setas: Comestibles vs. Venenosas con Machine Learning

Este proyecto implementa un flujo de trabajo completo de Ciencia de Datos para predecir si una seta es **comestible (`e`)** o **venenosa (`p`)** basándose en sus características morfológicas y de hábitat. 

Se evalúan múltiples algoritmos de clasificación y se optimiza el mejor modelo utilizando la librería de optimización de hiperparámetros **Optuna**.

---

## 📊 Estructura del Proyecto

El proyecto se compone de dos fases principales divididas en notebooks:

1. **`01_limpieza_eda.ipynb` (Análisis Exploratorio y Limpieza):** * Carga y exploración inicial del conjunto de datos de hongos.
   * Análisis de variables categóricas mediante tablas de contingencia y pruebas de Chi-cuadrado.
   * Conclusión clave: El **olor** es la característica más determinante para predecir la peligrosidad de una seta.
   * Exportación del dataset limpio a `data/mushrooms_clean.csv`.

2. **`02_modelo_clasificacion.ipynb` (Modelado y Optimización):**
   * Partición estratificada de los datos (80% entrenamiento / 20% test).
   * Transformación de columnas categóricas mediante *One-Hot Encoding*.
   * Evaluación de modelos base (`LogisticRegression`, `RandomForest`, `GradientBoosting`, `XGBoost`, `DecisionTree`).
   * Optimización bayesiana de hiperparámetros para **XGBoost** mediante **Optuna**.
   * Exportación del modelo final entrenado a la carpeta `models/`.

---

## 🛠️ Stack Tecnológico Utilizado

* **Lenguaje:** Python 3.x
* **Manipulación de datos:** `pandas`, `numpy`
* **Análisis estadístico y Visualización:** `scipy` (Chi-cuadrado), `matplotlib`, `seaborn`
* **Modelado y Validación:** `scikit-learn`
* **Algoritmo Avanzado:** `xgboost`
* **Optimización:** `optuna`
* **Persistencia del modelo:** `joblib`

## 🔍 Análisis de las Variables Relevantes

El proyecto aborda la relevancia de las variables desde dos perspectivas fundamentales: el análisis estadístico clásico en la fase de exploración (EDA) y la importancia algorítmica tras el entrenamiento del modelo.

### 1. Análisis Estadístico (Prueba de Chi-cuadrado)
Dado que el dataset está compuesto íntegramente por variables cualitativas (categóricas), durante la fase de EDA (`01_limpieza_eda.ipynb`) se utilizaron **tablas de contingencia** y la **prueba de Chi-cuadrado de independencia ($\chi^2$)** para medir la asociación entre cada característica morfológica y la variable objetivo (`class`).

* **El Olor (`odor`):** Resultó ser la variable con el mayor nivel de asociación estadística. El análisis demostró patrones casi perfectos: por ejemplo, las setas con olor a almendra (`a`) o anise (`l`) son sistemáticamente comestibles, mientras que olores pungentes (`p`), fétidos (`f`) o a pescado (`y`) correlacionan directamente con setas venenosas.
* **Color de las Láminas (`gill-color`):** Mostró también una altísima dependencia, donde ciertos colores (como el buff/crema `b`) alertan casi siempre de toxicidad.

### 2. Importancia de Características del Modelo (Feature Importance)
En el notebook de modelado (`02_modelo_clasificacion.ipynb`), tras entrenar el clasificador final **XGBoost** y aplicar *One-Hot Encoding*, se extrajo la importancia de las variables basada en la ganancia del modelo. 

Al agrupar las categorías codificadas de vuelta a sus variables originales, el top de características más influyentes para el modelo coincide plenamente con los hallazgos estadísticos previos:

1. **`odor` (Olor):** Concentra la mayor parte del peso predictivo del modelo.
2. **`gill-size` (Tamaño de las láminas):** Las láminas estrechas (`n`) presentan una alta frecuencia en ejemplares venenosos.
3. **`gill-color` (Color de las láminas):** Actúa como un fuerte validador secundario para el algoritmo.
4. **`spore-print-color` (Color de la espora):** Aporta información crucial en casos donde el olor es ausente o neutro.

> 💡 **Conclusión del Análisis:** No es necesario evaluar la totalidad de los caracteres morfológicos de una seta para determinar su peligro de forma inequívoca; basta con analizar un subconjunto crítico liderado de forma indiscutible por el **olor**.

---

## 📈 Resultados Obtenidos

### 1. Comparación de Modelos Base (Accuracy)
Todos los modelos alcanzan una precisión perfecta en el conjunto de test tras la codificación categórica:

| Modelo | Train Accuracy | Test Accuracy | Overfitting |
| :--- | :---: | :---: | :---: |
| **LogReg** | 1.0 | 1.0 | 0.0 |
| **RandomForest** | 1.0 | 1.0 | 0.0 |
| **GradientBoosting** | 1.0 | 1.0 | 0.0 |
| **XGBoost** | 1.0 | 1.0 | 0.0 |
| **DecisionTree** | 1.0 | 1.0 | 0.0 |

### 2. Validación Cruzada (Media F1-Macro)
Para asegurar que no existe un sobreajuste por la partición de los datos, la validación cruzada ratifica la excelente capacidad predictiva del dataset:
* **RandomForest / XGBoost:** 1.000000
* **GradientBoosting / DecisionTree:** 0.999692
* **Logistic Regression:** 0.999075

### 3. Optimización de Hiperparámetros (XGBoost)
Se ejecutó un estudio de 10 iteraciones con Optuna para encontrar la configuración óptima reduciendo la complejidad del modelo:

* **Mejor Valor F1:** 1.0
* **Parámetros óptimos encontrados:**
  ```python
  {
      'n_estimators': 96, 
      'max_depth': 5, 
      'learning_rate': 0.1459, 
      'subsample': 0.7572, 
      'colsample_bytree': 0.8245
  }

### 4. Reporte Final en el Conjunto de Test

El modelo final demostró un desempeño impecable identificando de manera inequívoca las muestras del conjunto de prueba (1625 setas analizadas):

```text
              precision    recall  f1-score   support

  Comestible       1.00      1.00      1.00       842
    Venenosa       1.00      1.00      1.00       783

    accuracy                           1.00      1625
   macro avg       1.00      1.00      1.00      1625
weighted avg       1.00      1.00      1.00      1625 
```

## 💡 Conclusión

Este proyecto demuestra con éxito la aplicación de un flujo de trabajo de Machine Learning robusto para resolver un problema de clasificación binaria crítica. Tras el análisis y modelado, se destacan las siguientes conclusiones clave:

1. **Determinación Biológica:** Mediante el Análisis Exploratorio de Datos (EDA) y las pruebas estadísticas de Chi-cuadrado, se identificó que el **olor** es la característica morfológica más determinante y con mayor poder predictivo para diferenciar setas comestibles de venenosas.
2. **Eficacia del Modelo:** El uso de algoritmos basados en árboles de decisión de gradiente mejorado (**XGBoost**) demostró una capacidad excepcional para capturar los patrones del dataset. Al aplicar codificación *One-Hot Encoding*, el modelo alcanzó una precisión e identificabilidad perfectas ($F1\text{-Score} = 1.00$) en el conjunto de test independiente, sin presentar signos de sobreajuste (overfitting).
3. **Eficiencia mediante Optimización:** La integración de **Optuna** permitió realizar una búsqueda bayesiana inteligente de hiperparámetros. Esto no solo aseguró el rendimiento óptimo del modelo, sino que ayudó a simplificar su estructura (controlando la profundidad máxima y submuestreo), garantizando un modelo final ligero, eficiente y altamente generalizable.

---
## 🚀 Cómo Ejecutar el Proyecto

## 1. Clonar el repositorio

Clona el proyecto en tu máquina local:

```bash
git clone https://github.com/Bootcamp-DA-P2/DA-project-classification-Grupo-1.git
```

Accede a la carpeta del proyecto:

```bash
cd DA-project-classification-Grupo-1
```

## 2. Cambiar a la rama principal

Asegúrate de estar trabajando en la rama principal del proyecto:

```bash
git switch main
```

o alternativamente:

```bash
git checkout main
```

## 3. Descargar el Dataset

Crear la carpeta:

```text
data/
```

Descargar el dataset desde:

https://archive.ics.uci.edu/dataset/73/mushroom

Utilizar el archivo:

```text
agaricus-lepiota.data
```

Una vez descargado, el archivo debe colocarse dentro de la carpeta data/ del proyecto para poder ser cargado correctamente desde el código.

---
## 4. Crear un entorno virtual

```bash
python -m venv venv
```

### Activación

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```
---
## 5.   Instala las dependencias requeridas:
```bash
    pip install -r requirements.txt
```
---
## 6. Generar el dataset procesado

Ejecutar:

```text
01_limpieza_eda.ipynb
```

Se generará:

```text
mushrooms_clean.csv
```
---
## 7. Entrenar los modelos

Ejecutar:

```text
02_modelo_clasificacion.ipynb
```

Se generará automáticamente:

```text
models/
```

Con los modelos serializados:

```text
best_xgboost_model.pkl
```
---
## 8. Ejecuta secuencialmente los notebooks 01_limpieza_eda.ipynb y 02_modelo_clasificacion.ipynb.

---   
## 9. Ejecutar la aplicación

```bash
streamlit run app/app.py
```
---
# 📁 Estructura del Proyecto

```text
.
├── app/
│   └── app.py
│
├── data/
│   ├── agaricus-lepiota.data
│   └── mushrooms_clean.csv
├── models/
│   ├ best_xgboost_model.pkl
│  
├── notebooks/
│   ├── 01_limpieza_eda.ipynb 
│   └── 02_modelo_clasificacion.ipynb.
│
├── requirements.txt
└── README.md
```
---

## 📊 Uso de dataset secundario (UCI - Secondary Mushroom Dataset)

Se ha utilizado un dataset secundario procedente del repositorio UCI Machine Learning Repository (Secondary Mushroom Dataset), disponible en:
[Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

Este dataset contiene una versión ampliada y más actual del conjunto clásico de setas, con aproximadamente 61.000 registros.

El archivo utilizado dentro del dataset es:

- `secondary_data.csv`

Este archivo debe colocarse en la carpeta `data` del proyecto para su correcta ejecución.

El trabajo con este dataset se ha realizado en la rama `feature/mushrooms-extended` del repositorio, donde se ha utilizado de forma experimental para comparar su comportamiento frente al dataset principal.

En dicha rama se han ejecutado los notebooks `01_limpieza_eda.ipynb`, que genera (`mushrooms_clean_secondary.csv`) en la carepta `data`, y `02_modelo_clasificacion`, generando un modelo optimizado mediante Optuna (`best_random_forest_optuna.pkl`), el cual se guarda en la carpeta `models`.

Sin embargo, este modelo no se ha incluido en la aplicación final, ya que su rendimiento es muy similar al modelo principal, pero con un mayor coste computacional debido al tamaño del dataset.

Por este motivo, se ha priorizado el modelo más eficiente para el despliegue final del proyecto.

--- 

## 📋 Gestión del Proyecto

Para la organización y seguimiento del trabajo se ha utilizado un tablero Kanban, permitiendo gestionar tareas, asignar responsabilidades y monitorizar el avance del proyecto de forma colaborativa.

🔗 [Tablero Kanban del proyecto](https://github.com/orgs/Bootcamp-DA-P2/projects/38)

---

## 🎬 Presentación del proyecto

La presentación visual del proyecto se ha realizado mediante Canva, donde se resume el objetivo, el proceso de desarrollo y los resultados del modelo de clasificación de setas.

📎 [Ver presentación del proyecto](https://canva.link/y9tcylu0roxejxw)

---

# 👥 Autores

* Alejandra Duque García
* José Carlos de Santiago Sánchez
* María Zorayda Bejarano Jilon
* Elena Suárez Serrano
