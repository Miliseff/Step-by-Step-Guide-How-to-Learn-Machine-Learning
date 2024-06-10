Cómo se utiliza EDA en el aprendizaje automático?

Como mencionamos, el enfoque de análisis de datos exploratorio no impone modelos deterministas o probabilísticos sobre los datos.

De lo contrario; El enfoque EDA permite que los datos sugieran modelos admisibles que mejor se ajusten a los datos.

Para el análisis exploratorio de datos, la atención se centra en los datos, su estructura, los valores atípicos y los modelos sugeridos por los datos.

Aunque existen otros métodos, el análisis de datos exploratorio generalmente se realiza utilizando los siguientes métodos.

El análisis univariante es la forma más sencilla de analizar datos; "uni" significa uno, en otras palabras, sus datos tienen solo una variable.

No trata causas o relaciones, a diferencia de la regresión, y su objetivo principal es describir.

Toma los datos, los resume y encuentra patrones en los datos.

En este ejemplo, verá dos tipos de datos univariados, categóricos y continuos.

Con el tipo de característica categórica, puede realizar EDA numérico usando la función de tabla cruzada de Pandas, y puede realizar EDA visual usando la función de gráfico de conteo de Seaborn.

Con el tipo de característica continua, puede realizar EDA numérica utilizando la función de descripción de Pandas y puede visualizar

diagramas de caja, diagramas de distribución y diagramas de estimación de densidad del kernel, o diagramas de KDE en Python, usando Matplotlib o Seaborn.

Hay muchas herramientas EDA a su disposición, pero eso está más allá del alcance de esta lección.


Puede utilizar la función Countplot de Seaborn para contar el número de observaciones en cada categoría.

Nuestra visualización es un gráfico de barras simple.

Análisis bivariado significa el análisis de datos bivariados.

Es una de las formas más simples de análisis estadístico y se utiliza para descubrir si existe una relación entre dos conjuntos de valores.

Por lo general, involucra las variables X e Y. Podemos analizar datos bivariados y datos multivariados en Python, usando Matplotlib o Seaborn, y también existen otras herramientas.

Una de las características más poderosas de Seaborn es la capacidad de crear fácilmente gráficos condicionales.

Esto nos permite ver cómo se ven los datos cuando se segmentan por una o más variables.

La forma más sencilla de hacerlo es mediante el método del gráfico de factores, que se utiliza para dibujar un gráfico categórico hasta una cuadrícula de facetas.

La función de gráfico conjunto de Seaborn dibuja un gráfico de dos variables con gráficos bivariados y univariados.

El método de mapa de diagrama de factores de Seaborn puede asignar un diagrama de factores a un gráfico de KDE, de distribución o de diagrama de caja.

Una gráfica común de datos bivariados es la gráfica lineal simple.

En este ejemplo, utilizamos la función regplot de Seaborn para visualizar una relación lineal entre dos conjuntos de características.

En este caso, la distancia del viaje, nuestra etiqueta X, y la cantidad justa, nuestro objetivo, parecen tener una relación lineal.

Tenga en cuenta que, aunque la mayoría de los datos tienden a agruparse de forma lineal, también hay valores atípicos.
