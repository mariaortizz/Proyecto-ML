# Proyecto Machine Learning - Viviendas

![imagen](./src/data/giphy.gif)

#### María del Rosario Ortiz
22 de febrero 2024

Se trata de un modelo de regresión para la predicción de precios por metro cuadrado de viviendas en la ciudad de Madrid. Esto nos permite tener un valor estimado previo a la tasación de un inmueble. Los principales beneficios que podemos ver obteniendo una valoración previa mediante un modelo de predicción son:
- Precisión
- Reconocimiento de características importantes
- Reducir el error o sesgo
- Eficiencia
- Manejo de grandes volúmenes de datos
- Automatización

Este trabajo de investigación comienza teniendo como objetivo analizar cómo se ve afectado el precio por metro cuadrado (€/m2) de la vivienda en Madrid, según distintas variables o características propias de estas. Posterior a este análisis que nos arroja las principales variables que están correlacionadas con nuestra variable target, se aplica una clusterización para poder agrupar los datos y reducir el margen de error. Por último, se implementa un modelo de regresión para la predicción de estos valores.

Los datos utilizados se obtuvieron de [Link Dataset](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market), que

#### Contiene información relativa a:

- annio_construcción: año de construcción del inmueble
- precio_venta: valor de la inmueble
- precio_venta_por_m2: valor de la inmueble por metro cuadrado
- cee: certificado energético del inmueble
- piso: planta a la que se encuentra
- balcón: si tiene o no tiene balcón
- armarios_empotrados: si tiene o no tiene armarios empotrados
- jardin: si tiene o no tiene jardín
- zonas_verdes: si tiene o no tiene zonas verdes
- estacionamiento: si tiene o no tiene estacionamiento
- piscina: si tiene o no tiene piscina
- trastero: si tiene o no tiene trastero
- terraza: si tiene o no tiene terraza
- tipo_inmueble: si es un piso, un dúplex, chalet o ático
- accesible: si es o no accesible para personas con movilidad reducida
- exterior: si tiene o no tiene exterior
- bajo: si es o no es un piso en planta baja
- necesita_reforma: si la vivienda necesita o no necesita reforma
- bannos: cantidad de baños del inmueble
- habitaciones: cantidad de habitaciones del inmueble
- metros_cuadrados: cuántos metros cuadrados tiene el inmueble
- ubicación: donde se ubica geográficamente el inmueble

#### Métodos:

Mediante métodos gráficos y estadísticos se analiza la relación entre la variable principal (precio_venta_por_m2) contra las demás variables del dataset. Se implementa un modelo de clusterización (KMeans) para poder agrupar los datos y por último se utiliza Ensemble Learning (Aprendizaje en conjunto) para predecir los valores finales.

#### Resultados hacia futuro:

En resumen, una valoración previa es fundamental para obtener una estimación inicial del valor de un activo antes de realizar una tasación final, y utilizar un modelo de Machine Learning puede mejorar la precisión, eficiencia y capacidad para identificar factores importantes en este proceso.