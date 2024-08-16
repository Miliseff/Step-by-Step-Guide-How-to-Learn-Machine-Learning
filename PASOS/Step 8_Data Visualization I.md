# # Data Visualization I

# Types of Charts and Their Selection

## 1. Types of Charts

### 1.1. Distribution
Distribution charts are used to show how data is distributed across a range of values. This includes histograms, density plots, and box plots.

#### 1.1.1. Histogram
A histogram shows the distribution of a continuous variable by dividing the data into intervals (bins) and counting the frequency of data within each interval.


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

#### 1.1.2. Density Plot
A density plot is similar to a histogram but uses a kernel density function to estimate the distribution of the variable.

```python
import seaborn as sns

# Crear el gráfico de densidad
sns.kdeplot(data, shade=True)
plt.title('Gráfico de Densidad')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.show()
```

#### 1.1.3. Box Plot
A box plot shows the distribution of the data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum.

```python
# Crear el diagrama de caja
plt.boxplot(data, vert=False)
plt.title('Diagrama de Caja')
plt.xlabel('Valor')
plt.show()
```

### 1.2. Ranking
Ranking charts are used to show comparisons between different categories. This includes bar charts and pie charts.

#### 1.2.1. Bar Chart
A bar chart displays categorical data with rectangles whose height or length is proportional to the values they represent.

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

#### 1.2.2. Pie Chart
A pie chart shows the proportion of each category as a segment of a circle.

```python
# Crear el gráfico de torta
plt.pie(values, labels=categories, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title('Gráfico de Torta')
plt.show()
```

### 1.3. Correlation
Correlation charts are used to show the relationship between two variables. This includes scatter plots and 2D density plots.

#### 1.3.1. Scatter Plot
A scatter plot shows the values of two variables as points on a Cartesian plane, allowing observation of the relationship between them.

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

#### 1.3.2. 2D Density Plot
A 2D density plot is similar to a scatter plot but uses colors to indicate the density of points in each area of the plot.

```python
# Crear el gráfico de densidad 2D
sns.kdeplot(x, y, cmap="Blues", shade=True, bw_adjust=.5)
plt.title('Gráfico de Densidad 2D')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```


## 2. Unidimensional and Bidimensional Charts
Unidimensional Charts: Show the distribution of a single variable (e.g., histograms, density plots, box plots).
Bidimensional Charts: Show the relationship between two variables (e.g., scatter plots, 2D density plots).

## 3. Chart Selection
The selection of the appropriate chart depends on the variables being analyzed and the message that needs to be communicated:

Distribution: Use histograms, density plots, or box plots to show how data is distributed.
Ranking: Use bar charts or pie charts to compare different categories.
Correlation: Use scatter plots or 2D density plots to explore the relationship between two variables.

When choosing a chart, it is crucial to consider the type of data, the number of variables involved, and the nature of the information you want to convey. This ensures that the selected chart clearly communicates the intended message.
