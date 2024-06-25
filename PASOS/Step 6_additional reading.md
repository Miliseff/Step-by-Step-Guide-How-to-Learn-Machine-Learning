# Cómo se utiliza EDA en el aprendizaje automático?

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

>> En este módulo cubriremos el aprendizaje supervisado, que es una rama del aprendizaje automático en la que se dan ejemplos etiquetados del modelo.

Dos de las clases más comunes de modelos de aprendizaje automático son los modelos de ML supervisados ​​y no supervisados.

La diferencia clave es que con los modelos supervisados, tenemos etiquetas o, en otras palabras, las respuestas correctas a lo que sea que debemos aprender a predecir.

En el aprendizaje no supervisado, los datos no tienen etiquetas.

Este gráfico es un ejemplo de un tipo de problema que un modelo no supervisado podría intentar resolver.

Aquí queremos analizar la antigüedad y los ingresos y luego agrupar o agrupar a los empleados para ver si alguien está en la vía rápida.

Fundamentalmente, aquí no hay ninguna verdad fundamental.

La gerencia no tiene, hasta donde sabemos, una gran mesa de personas a las que van a promover rápidamente y aquellas a las que no van a promover.

En consecuencia, los problemas no supervisados ​​tienen que ver con el descubrimiento, con mirar los datos sin procesar y ver si se dividen naturalmente en grupos.

A primera vista parece que hay dos clusters o grupos distintos que podría separar muy bien con una línea.

Sin embargo, en este curso nos centraremos en problemas de aprendizaje automático supervisado como este.

La diferencia fundamental es que con el aprendizaje supervisado tenemos cierta noción de una etiqueta o una característica de cada punto de datos que nos importa mucho.

Normalmente, esto es algo que conocemos a través de datos históricos pero no lo sabemos en tiempo real.

Sabemos otras cosas que llamamos predictores y queremos utilizar esos predictores para predecir lo que no sabemos.

Por ejemplo, digamos que eres el camarero de un restaurante.

Tienes datos históricos del monto de la factura y de cuánta propina dieron las diferentes personas.

Ahora estás mirando al grupo sentado en la mesa de la esquina.

Sabes cuál es su factura total, pero no sabes cuál será su propina.

En datos históricos, la propina es una etiqueta.

Usted crea un modelo para predecir la propina a partir del monto de la factura, luego intenta predecir el

Sugerencia en tiempo real basada en los datos históricos y los valores que conoce para la tabla específica.

Dentro del ML supervisado, existen dos tipos de problemas: regresión y clasificación.

Para explicarlos, profundicemos un poco más en estos datos.

En este conjunto de datos de propinas, un conjunto de datos de ejemplo que viene con un paquete de Python seaborn, cada fila tiene muchas características, como factura total, propina y sexo.

En el aprendizaje automático, llamamos a cada fila un ejemplo.

Elegiremos una de las columnas como la característica que queremos predecir llamada etiqueta, y elegiremos un conjunto de otras columnas que se denominan características.

En la opción uno del modelo, queremos predecir el monto de la propina; por lo tanto, la punta de la columna es mi etiqueta.

Puedo usar una, todas o cualquier cantidad de columnas como funciones para predecir la propina.

Este será un modelo de regresión porque la punta es una etiqueta continua.

En la opción dos del modelo, queremos predecir el sexo del cliente; por lo tanto, el sexo de la columna es la etiqueta.

Una vez más, usaré un conjunto del resto de columnas como funciones para intentar predecir el sexo del cliente.

Este será un modelo de clasificación porque nuestra etiqueta sexo tiene un número discreto de valores o clases.

En resumen, dependiendo del problema que intente resolver, los datos que tenga, la explicabilidad, etc., determinarán qué modelos de aprendizaje automático utilizará para encontrar una solución.

Sus datos no están etiquetados, no podremos utilizar el aprendizaje supervisado en ese momento y recurriremos a algoritmos de agrupación para descubrir propiedades interesantes de los datos.

Tus datos están etiquetados y la etiqueta es raza de perro, que es una cantidad discreta ya que hay un número finito de razas de perros, utilizamos un algoritmo de clasificación entonces.

Si en cambio la etiqueta es peso del perro, que es una cantidad continua, deberíamos utilizar un algoritmo de regresión.

La etiqueta, nuevamente, es lo que estás tratando de predecir.

En el aprendizaje supervisado, tienes algunos datos con las respuestas correctas.
