Step 1 - Introduccion a Data Science 

Subject- Tus primeros pasos
-1 Importación de las Bibliotecas Requeridas
Estas dos bibliotecas son esenciales y las importaremos cada vez.
NumPy es una biblioteca que contiene funciones matemáticas.
Pandas es la biblioteca utilizada para importar y gestionar los conjuntos de datos.

-2: Importación del Conjunto de Datos
Los conjuntos de datos suelen estar disponibles en formato .csv. Un archivo CSV almacena datos tabulares en texto plano. Cada línea del archivo es un registro de datos. Utilizamos el método read_csv de la biblioteca pandas para leer un archivo CSV local como un dataframe. Luego creamos una matriz y un vector separados de variables independientes y dependientes a partir del dataframe.

-3: Manejo de Datos Faltantes
Los datos que obtenemos rara vez son homogéneos. Los datos pueden faltar debido a diversas razones y es necesario manejarlos para que no reduzcan el rendimiento de nuestro modelo de aprendizaje automático. Podemos reemplazar los datos faltantes por la media o la mediana de toda la columna. Utilizamos la clase Imputer de sklearn.preprocessing para esta tarea.

-4: Codificación de Datos Categóricos
Los datos categóricos son variables que contienen valores de etiqueta en lugar de valores numéricos. El número de valores posibles suele estar limitado a un conjunto fijo. Ejemplos de valores como "Sí" y "No" no se pueden usar en ecuaciones matemáticas del modelo, por lo que necesitamos codificar estas variables en números. Para lograr esto, importamos la clase LabelEncoder de la biblioteca sklearn.preprocessing.

-5: División del conjunto de datos en conjunto de prueba y conjunto de entrenamiento
Creamos dos particiones del conjunto de datos, una para entrenar el modelo llamada conjunto de entrenamiento y otra para probar el rendimiento del modelo entrenado llamada conjunto de prueba. La división es generalmente 80/20. Importamos el método train_test_split() de la biblioteca sklearn.crossvalidation.

-6: Escalado de Características
La mayoría de los algoritmos de aprendizaje automático utilizan la distancia euclidiana entre dos puntos de datos en sus cálculos, las características altamente variables en magnitudes, unidades y rango plantean problemas. Las características de magnitudes altas pesarán más en los cálculos de distancia que las características con magnitudes bajas. Se realiza mediante estandarización de características o normalización de puntuación Z. Se importa StandardScaler de sklearn.preprocessing.


Subject- Regresion Lineal Simple

Subject- Regresion Lineal Multiple

Subject- Regresion Logistica

Subject- Regresion Lineal Multiple--5-6

Subject- K Nearest Neighbours



Subject- SVM (support vector machines)

Subjet- K-NN 

Clasificador Bayes- Black Box Machine Learning--13--15

Implementacion SVM utilizando Scikit-Learn--14

Implementacion SVM utilizando Kernel Trick--16

-------------Mejora de las redes neuronales profundas: ajuste, regularización y optimización de hiperparámetros. -- posible titulo ?












