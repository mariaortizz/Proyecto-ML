import pandas as pd

# Supongamos que 'datos_entrenamiento' es tu DataFrame original usado para entrenar el modelo
# Y 'nuevos_datos_df' es tu nuevo DataFrame con los datos a predecir

# 1. Asegúrate de que 'nuevos_datos_df' tenga las mismas columnas que 'datos_entrenamiento'
columnas_nuevos_datos = datos_entrenamiento.columns
nuevos_datos_df = nuevos_datos_df[columnas_nuevos_datos]

# 2. Convierte 'nuevos_datos_df' en un arreglo NumPy
nuevos_datos = nuevos_datos_df.values

# 3. Realiza predicciones con el modelo KMeans
nuevas_predicciones = kmeans.predict(nuevos_datos)

print("Predicciones para los nuevos datos:", nuevas_predicciones)
