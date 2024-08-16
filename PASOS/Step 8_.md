# Resumen de Visualización de Datos I

- La visualización de datos facilita la interpretación y comunicación de información.
- Tipos de gráficos:
  - **Distribución**: Histogramas, gráficos de densidad, diagramas de caja.
  - **Ranking**: Gráficos de barras, gráficos de torta.
  - **Correlación**: Gráficos de dispersión, gráficos de densidad 2D.
- Gráficos unidimensionales muestran una variable; bidimensionales muestran relaciones entre dos variables.
- Selección de gráficos depende de las variables y el mensaje a comunicar.


Ventajas de Gráficos de Densidad sobre Histogramas
Los gráficos de densidad proporcionan una representación más suave y continua de la distribución de datos, lo que facilita la identificación de patrones y tendencias.

Permiten comparar múltiples distribuciones en un solo gráfico sin la limitación de los intervalos de clase que tienen los histogramas.

La suma de las áreas bajo la curva de un gráfico de densidad es igual a 1, lo que facilita la interpretación de probabilidades.



# Tipos de Gráficos y su Selección

## 1. Tipos de Gráficos

### 1.1. Distribución
Los gráficos de distribución se utilizan para mostrar cómo los datos se distribuyen a lo largo de un rango de valores. Aquí se incluyen los histogramas, gráficos de densidad y diagramas de caja.

#### 1.1.1. Histograma
Un histograma muestra la distribución de una variable continua dividiendo los datos en intervalos (bins) y contando la frecuencia de los datos en cada intervalo.

```python
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
data = np.random.normal(0, 1, 1000)

# Crear el histograma
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histograma de Distribución')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()
```

#### 1.1.2. Gráfico de Densidad
Un gráfico de densidad es similar a un histograma, pero utiliza una función de densidad de kernel para estimar la distribución de la variable.

```python
import seaborn as sns

# Crear el gráfico de densidad
sns.kdeplot(data, shade=True)
plt.title('Gráfico de Densidad')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.show()
```

#### 1.1.3. Diagrama de Caja
Un diagrama de caja muestra la distribución de los datos basado en un resumen de cinco números: mínimo, primer cuartil (Q1), mediana, tercer cuartil (Q3), y máximo.

```python
# Crear el diagrama de caja
plt.boxplot(data, vert=False)
plt.title('Diagrama de Caja')
plt.xlabel('Valor')
plt.show()
```

### 1.2. Ranking
Los gráficos de ranking se utilizan para mostrar comparaciones entre diferentes categorías. Aquí se incluyen los gráficos de barras y gráficos de torta.

#### 1.2.1. Gráfico de Barras
Un gráfico de barras muestra datos categóricos con rectángulos cuya altura o longitud es proporcional a los valores que representan.

```python
categories = ['A', 'B', 'C', 'D']
values = [5, 7, 3, 8]

# Crear el gráfico de barras
plt.bar(categories, values, color='skyblue')
plt.title('Gráfico de Barras')
plt.xlabel('Categoría')
plt.ylabel('Valor')
plt.show()
```

#### 1.2.2. Gráfico de Torta
Un gráfico de torta muestra la proporción de cada categoría como un segmento de un círculo.

```python
# Crear el gráfico de torta
plt.pie(values, labels=categories, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title('Gráfico de Torta')
plt.show()
```

### 1.3. Correlación
Los gráficos de correlación se utilizan para mostrar la relación entre dos variables. Aquí se incluyen los gráficos de dispersión y gráficos de densidad 2D.

#### 1.3.1. Gráfico de Dispersión
Un gráfico de dispersión muestra los valores de dos variables como puntos en un plano cartesiano, lo que permite observar la relación entre ellas.

```python
x = np.random.rand(100)
y = x + np.random.normal(0, 0.1, 100)

# Crear el gráfico de dispersión
plt.scatter(x, y, alpha=0.5)
plt.title('Gráfico de Dispersión')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```

#### 1.3.2. Gráfico de Densidad 2D
Un gráfico de densidad 2D es similar a un gráfico de dispersión, pero utiliza colores para indicar la densidad de puntos en cada área del gráfico.

```python
# Crear el gráfico de densidad 2D
sns.kdeplot(x, y, cmap="Blues", shade=True, bw_adjust=.5)
plt.title('Gráfico de Densidad 2D')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```


## 2. Gráficos Unidimensionales y Bidimensionales
Gráficos Unidimensionales: Muestran la distribución de una única variable (ej. histogramas, gráficos de densidad, diagramas de caja).
Gráficos Bidimensionales: Muestran la relación entre dos variables (ej. gráficos de dispersión, gráficos de densidad 2D).

## 3. Selección de Gráficos
La selección del gráfico adecuado depende de las variables que se están analizando y del mensaje que se desea comunicar:

Distribución: Utiliza histogramas, gráficos de densidad o diagramas de caja para mostrar cómo se distribuyen los datos.
Ranking: Usa gráficos de barras o gráficos de torta para comparar diferentes categorías.
Correlación: Utiliza gráficos de dispersión o gráficos de densidad 2D para explorar la relación entre dos variables.
Al elegir un gráfico, es crucial considerar el tipo de datos, el número de variables involucradas y la naturaleza de la información que se quiere transmitir. Esto asegura que el gráfico seleccionado comunique claramente el mensaje deseado.