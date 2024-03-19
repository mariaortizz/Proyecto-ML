#librerias EDA/estandar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, probplot, shapiro, spearmanr, kruskal
from geopy.geocoders import Nominatim
import time
import plotly.express as px
import traceback

#librerias modelos
from sklearn.preprocessing import OrdinalEncoder

pd.set_option('display.max_columns', None)


def nombre_columnas(data):
    '''Función para tratar el nombre de las columnas y eliminar las vacias'''
    try:
        data.drop(columns=['latitude', 'longitude', 'portal', 'door', 'rent_price_by_area', 'are_pets_allowed', 'is_furnished',
                    'is_kitchen_equipped', 'has_private_parking', 'has_public_parking', 'sq_mt_useful', 'n_floors', 'has_ac', 'title',
                    'sq_mt_allotment', 'raw_address', 'is_exact_address_hidden', 'street_name', 'street_number', 'is_buy_price_known',
                    'is_parking_included_in_price', 'is_rent_price_known', 'operation', 'is_new_development', 'parking_price', 'rent_price', 'id', 'neighborhood_id',
                    'has_central_heating', 'has_individual_heating', 'has_lift', 'is_orientation_east', 'is_orientation_north', 'is_orientation_south', 'is_orientation_west'
                    ], axis=1, inplace = True)
        
        data.columns = ['annio_construccion', 'precio_venta', 'precio_venta_por_m2', 'cee',
    'piso', 'balcon', 'armarios_empotrados', 'jardin', 'zonas_verdes', 
    'estacionamiento', 'pileta',
    'trastero', 'terraza', 'tipo_inmueble',
    'accesible', 'exterior', 'bajo', 'necesita_reforma', 'bannos', 'habitaciones',
        'metros_cuadrados', 'ubicacion']

    except Exception as a:
        print(f"No pude tranformar las columnas por {a}")
    return data

def cardinalidad(data):
    '''Funcion para saber la cardinalidad de las varibales que tenemos en el data frame'''
    df_cardin = pd.DataFrame([{
                'variable' : i,
                'tipo_dato' : data[i].dtypes,
                'cantidad_de_nulos' : data[i].isna().sum(),
                'valores_unicos' : data[i].unique(),
                'cardinalidad' : data[i].nunique(),
                'porcentaje_cardinalidad' : (data[i].nunique()/data.shape[0])*100
            } for i in data])
    
    # df_tipo_variable = pd.DataFrame({'tipo_variable' : ['discreta', 'continua', 'continua', 'ordinal',
    #         'ordinal', 'nominal', 'nominal', 'nominal', 'nominal',
    #         'nominal', 'nominal', 'nominal', 'nominal', 'nominal',
    #         'nominal', 'nominal', 'nominal', 'nominal', 'discreta',
    #         'discreta', 'continua', 'nominal']})
    
    # df_cardinalidad = pd.concat([df_cardin,df_tipo_variable], axis = 1)

    return df_cardin

def graficos_variables_cuant(data):
    ''''Funcuion para graficar las variables cuantitativas'''
    media_color = 'r'
    mediana_color = 'b'
    try:
        for columna in data.columns:
            print('--'*30)
            print(f"VARIABLE: {columna}\n")

            media = data[columna].mean()
            mediana = data[columna].median()

            plt.figure(figsize=(20,4))
            sns.boxplot(data[columna], orient='h', palette='husl')
            plt.axvline(media, color = media_color, linestyle = 'dashed', linewidth = 1)
            plt.axvline(mediana, color = mediana_color, linestyle = 'dashed', linewidth = 1)

            plt.show()

            sns.displot(data[columna], rug = True, palette='husl' , bins = 30)
            plt.axvline(media, color = media_color, linestyle = 'dashed', linewidth = 1, label = f'Media: {media:.0f}')
            plt.axvline(mediana, color = mediana_color, linestyle = 'dashed', linewidth = 1, label = f'Mediana: {mediana:.0f}')

            plt.title(f'Distribución de {columna}')
            plt.legend()

            plt.show()

            print(data[columna].describe().round())
            print('--'*30)
    except Exception as a:
        print(f"No puedo analizar la variable por este error {a}")

def graficos_variables_cualit(data):
    ''''Funcuion para graficar las variables cualitativas'''
    try:
        for columna in data.columns:
            print('--'*50)
            print(f"VARIABLE: {columna}\n")
            if len(data[columna].dropna().unique()) > 1:
                ax = sns.countplot(data= data.sort_values(by=columna), x= columna, palette='husl')
                ax.set_xticklabels(data[columna].sort_values().unique(), rotation=90)
                #se superponen los valores porque me da uns advertencia al aplicar este parametro, no sé como sacarla :)
                plt.title(f"Conteo variable {columna}")
                plt.show()
            else:
                print('No es necesario graficar porque tiene un solo valor dentro de la columna')
            print(data[columna].value_counts())
            print('--'*50)
    except Exception as a:
        print(f"No puedo analizar la variable por este error {a}")

def rellenar_columnas_F(data):
    ''' Función que rellena las columnas que tienen valor true y nan'''
    try:
        data['zonas_verdes'].replace(np.nan, False,inplace=True)
        data['balcon'].replace(np.nan, False,inplace=True)
        data['armarios_empotrados'].replace(np.nan, False,inplace=True)
        data['jardin'].replace(np.nan, False,inplace=True)
        data['pileta'].replace(np.nan, False,inplace=True)
        data['trastero'].replace(np.nan, False,inplace=True)
        data['terraza'].replace(np.nan, False,inplace=True)
        data['accesible'].replace(np.nan, False,inplace=True)
    except Exception as a:
        print(f"No pude rellenar las columnas por {a}")
    return data

def rellenar_annios_nulos_necesitan_reforma(data):
    '''Función para rellenar los annios que vienen nulos'''
    try:
        #diccionario para ver si tenemos todos las ubicaciones o no
        dicc_annios_antiguos = data[(data['necesita_reforma']) & (data['annio_construccion'].notna())].sort_values(by = 'annio_construccion').groupby('ubicacion')[['annio_construccion']].median(numeric_only = True).astype(int).reset_index().to_dict('records')
        
        #agrega las ubicaciones que no existen, asignando la media de los inmuebles que necesitan reforma
        dicc_annios_antiguos = dicc_annios_antiguos + [{'ubicacion': 'Horcajo, Madrid', 'annio_construccion': 1957}, 
                   {'ubicacion': 'Valdebebas - Valdefuentes, Madrid', 'annio_construccion': 1957},
                   {'ubicacion': 'Virgen del Cortijo - Manoteras, Madrid', 'annio_construccion': 1957}]
        
        data_annios_antiguos = pd.DataFrame(dicc_annios_antiguos)
        data_unido = pd.merge(data,data_annios_antiguos, on='ubicacion', how = 'left')

        #asigna el valor de el annio en base a la la ubicacion
        data_unido['annio_construccion'] = data_unido.apply(lambda x: x.annio_construccion_y if ((x.necesita_reforma) & (pd.isna(x.annio_construccion_x))) else x.annio_construccion_x, axis = 1)

        data = data_unido.drop(columns=['annio_construccion_y', 'annio_construccion_x'], axis = 1)
    
    except Exception as a:
        print(f"No pude transformar el df por {a}")

    return data

def rellenar_annios_nulos_no_necesitan_reforma(data):
    '''Función para rellenar los annios que vienen nulos'''
    try:
        #diccionario para ver si tenemos todos las ubicaciones o no
        dicc_annios_nuevo = data[(~data['necesita_reforma']) & (data['annio_construccion'].notna())].sort_values(by = 'annio_construccion').groupby('ubicacion')[['annio_construccion']].median(numeric_only = True).astype(int).reset_index().to_dict('records')
        
        #agrega las ubicaciones que no existen, asignando la media de los inmuebles que no necesitan reforma
        dicc_annios_nuevo = dicc_annios_nuevo + [{'ubicacion': 'Cuatro Vientos, Madrid', 'annio_construccion': 1973}]
        
        data_annios_nuevo = pd.DataFrame(dicc_annios_nuevo)
        data_unido_nuevo = pd.merge(data,data_annios_nuevo, on='ubicacion', how='left')

        #asigna el valor de el annio en base a la la ubicacion
        data_unido_nuevo['annio_construccion'] = data_unido_nuevo.apply(lambda x: x.annio_construccion_y if ((~x.necesita_reforma) & (pd.isna(x.annio_construccion_x))) else x.annio_construccion_x, axis = 1)

        data = data_unido_nuevo.drop(columns=['annio_construccion_x', 'annio_construccion_y'], axis = 1)
    
    except Exception as a:
        print(f"No pude transformar el df por {a}")

    return data

def rellenar_annio_outlier(data):
    '''Funcion para rellenar un año de construccion incorrecto'''
    # media_año_barrio_s = data[(data['ubicacion'] == 'Barrio de Salamanca, Madrid') & (data['annio_construccion'].notna())].sort_values(by = 'annio_construccion').groupby('ubicacion')['annio_construccion'].median(numeric_only = True).astype(int)
    data['annio_construccion'].replace(8170.0, 1947, inplace= True)
    return data

def rellenar_pisos_nulos(data):
    '''Funcion para rellenar los valores nulos de los pisos, con la moda segun la ubicacion'''
    try:
        #df el piso que más se repite, respetando las alturas por ubicacion segun normativa
        df_piso_más_comun = data[data['piso'].notna()].groupby(['ubicacion', 'piso'], as_index=False).count()[['ubicacion', 'piso']].groupby('ubicacion', as_index=False).max()

        df_unido_pisos = pd.merge(data,df_piso_más_comun, on='ubicacion', how= 'inner')

        df_unido_pisos['piso'] = df_unido_pisos.apply(lambda x: x.piso_y if pd.isna(x.piso_x) else x.piso_x, axis = 1)

        data = df_unido_pisos.drop(columns=['piso_x', 'piso_y'], axis = 1)
    except Exception as a:
        print(f"No pude transformar el df por {a}")
    return data

def rellenar_bajos_nulos(data):
    '''Funcion que rellena los valores nulos en la columna bajo en base al piso en el que se encuentra'''
    try:
        bajos = ('Semi-sótano', 'Entreplanta interior', 'Entreplanta', 'Semi-sótano exterior', 'Semi-sótano interior', 'Sótano interior', 'Sótano', 'Sótano exterior')

        data['bajo'] = data['piso'].apply(lambda x: True if x in bajos else False)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def sacar_metros_cuadrados_nuevos(data):
    ''''Funcion para rellenar los valores nulos de los metros cuadrados en base a el precio por metro cuadrado'''
    try:
        data.drop(columns=['metros_cuadrados'], axis= 1, inplace=True)
        data['metros_cuadrados'] = (data['precio_venta'] / data['precio_venta_por_m2']).round()
    except Exception:
        print("No pude tranformar el dataframe en la función 'sacar_metros_cuadrados_nuevos'")
    return data

def rellenar_exterior(data):
    '''Funcion que rellena los valores nulos en la columna exterior en base a el piso en el que se encuentra'''
    try:
        exteriores = ('Entreplanta exterior', 'Semi-sótano exterior', 'Sótano exterior')

        data['exterior'] = data['piso'].apply(lambda x: True if x in exteriores else False)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def rellenar_tipo_inmueble(data):
    '''Funcion que rellena los valores nulos en la columna tipo_inmueble, los unico no completos son los estudios'''
    try:
        data['tipo_inmueble'].fillna(value ='HouseType 1: Pisos', inplace=True)

    except Exception as a:
        print(f"No puse tranformar el df por {a}")
    
    return data

def rellenar_bannos_nulos(data):
    '''Funcion para rellenar los valores nulos de los bannos, con la media segun los metros cuadrados'''
    try:
        dicc_bannos = data[data['bannos'].notna()].groupby(['metros_cuadrados'], as_index=False)[['bannos']].mean().round().to_dict('records')
        dicc_bannos.append({'metros_cuadrados': 661, 'bannos': 5})

        df_banno_mas_comun = pd.DataFrame(dicc_bannos)

        df_unido_bannos = pd.merge(data,df_banno_mas_comun, on='metros_cuadrados', how= 'inner')

        df_unido_bannos['bannos'] = df_unido_bannos.apply(lambda x: x.bannos_y if pd.isna(x.bannos_x) else x.bannos_x, axis = 1)

        data = df_unido_bannos.drop(columns=['bannos_x', 'bannos_y'], axis = 1)
    except Exception as a:
        print(f"No pude transformar el df por {a}")
    return data

def zonas_nuevo(data):
    '''Funcion para subdivir las localizaciones por zonas'''
    
    centro = ['Centro, Madrid', 'Palacio, Madrid', 'Lavapiés-Embajadores, Madrid', 'Huertas-Cortes, Madrid', 'Chueca-Justicia, Madrid', 
            'Malasaña-Universidad, Madrid', 'Sol, Madrid']
    arganzuela = ['Arganzuela, Madrid', 'Imperial, Madrid', 'Acacias, Madrid', 'Chopera, Madrid', 'Legazpi, Madrid', 'Delicias, Madrid', 
                'Palos de Moguer, Madrid']
    retiro = ['Retiro, Madrid', 'Pacífico, Madrid', 'Adelfas, Madrid', 'Estrella, Madrid', 'Ibiza, Madrid', 'Jerónimos, Madrid', 'Niño Jesús, Madrid']
    salamanca = ['Barrio de Salamanca, Madrid', 'Recoletos, Madrid', 'Goya, Madrid', 'Fuente del Berro, Madrid', 'Guindalera, Madrid', 'Lista, Madrid', 'Castellana, Madrid']
    chamartin = ['Chamartín, Madrid','El Viso, Madrid', 'Prosperidad, Madrid', 'Ciudad Jardín, Madrid', 'Bernabéu-Hispanoamérica, Madrid', 
                'Nueva España, Madrid', 'Castilla, Madrid']
    tetuan = ['Tetuán, Madrid', 'Bellas Vistas, Madrid', 'Cuatro Caminos, Madrid', 'Cuzco-Castillejos, Madrid', 'Ventilla-Almenara, Madrid', 
            'Valdeacederas, Madrid', 'Berruguete, Madrid']
    chamberi = ['Nuevos Ministerios-Ríos Rosas, Madrid', 'Chamberí, Madrid','Gaztambide, Madrid', 'Arapiles, Madrid', 'Trafalgar, Madrid', 'Almagro, Madrid', 'Vallehermoso, Madrid']
    fuencarral_pardo = ['Montecarmelo, Madrid', 'Fuencarral, Madrid','El Pardo, Madrid', 'Fuentelarreina, Madrid', 'Peñagrande, Madrid', 'Pilar, Madrid', 'La Paz, Madrid', 
                        'Arroyo del Fresno, Madrid','Las Tablas, Madrid','Tres Olivos - Valverde, Madrid', 'Mirasierra, Madrid']
    moncloa_arav = ['Moncloa, Madrid', 'Casa de Campo, Madrid', 'Argüelles, Madrid', 'Ciudad Universitaria, Madrid', 'Valdezarza, Madrid', 
                    'Valdemarín, Madrid', 'El Plantío, Madrid', 'Aravaca, Madrid']
    latina = ['Latina, Madrid', 'Los Cármenes, Madrid', 'Puerta del Ángel, Madrid', 'Lucero, Madrid',
            'Águilas, Madrid','Aluche, Madrid','Campamento, Madrid','Cuatro Vientos, Madrid','Las Águilas, Madrid']
    carabanchel = ['Pau de Carabanchel, Madrid','Carabanchel, Madrid','Comillas, Madrid','Opañel, Madrid','San Isidro, Madrid',
                'Vista Alegre, Madrid','Puerta Bonita, Madrid','Buena Vista, Madrid','Abrantes, Madrid']
    usera = ['12 de Octubre-Orcasur, Madrid','Usera, Madrid','Orcasitas, Madrid','Opañel, Madrid','San Fermín, Madrid',
            'Almendrales, Madrid','Moscardó, Madrid','Zofío, Madrid','Pradolongo, Madrid']
    puente_vallecas = ['Puente de Vallecas, Madrid', 'Entrevías, Madrid','San Diego, Madrid','Palomeras Bajas, Madrid',
                    'Palomeras sureste, Madrid','Portazgo, Madrid','Numancia, Madrid']
    moratalaz = ['Moratalaz, Madrid', 'Pavones, Madrid', 'Horcajo, Madrid', 'Marroquina, Madrid', 'Media Legua, Madrid',
                'Fontarrón, Madrid', 'Vinateros, Madrid']
    ciudad_lineal = ['Ciudad Lineal, Madrid', 'Ventas, Madrid', 'Pueblo Nuevo, Madrid', 'Quintana, Madrid', 'Concepción, Madrid',
                    'San Pascual, Madrid', 'San Juan Bautista, Madrid', 'Colina, Madrid', 'Atalaya, Madrid', 'Costillares, Madrid']
    hortaleza = ['Sanchinarro, Madrid', 'Hortaleza, Madrid', 'Conde Orgaz-Piovera, Madrid', 'Palomas, Madrid', 'Canillas, Madrid', 'Pinar del Rey, Madrid',
                'Virgen del Cortijo - Manoteras, Madrid','Apóstol Santiago, Madrid', 'Valdebebas - Valdefuentes, Madrid']
    villaverde = ['San Andrés, Madrid','Villaverde, Madrid', 'San Cristóbal, Madrid', 'Butarque, Madrid', 'Los Rosales, Madrid', 'Los Ángeles, Madrid']
    villa_vallecas = ['Villa de Vallecas, Madrid','Casco Histórico de Vallecas, Madrid', 'Santa Eugenia, Madrid', 'Ensanche de Vallecas - La Gavia, Madrid']
    vicalvaro = ['Vicálvaro, Madrid','Casco Histórico de Vicálvaro, Madrid', 'Valdebernardo - Valderribas, Madrid', 'El Cañaveral - Los Berrocales, Madrid']
    barajas = ['Campo de las Naciones-Corralejos, Madrid','Barajas, Madrid','Ambroz, Madrid','Alameda de Osuna, Madrid', 'Casco Histórico de Barajas, Madrid', 'Timón, Madrid']
    

    try:
        funcion_lambda = lambda x: 'centro' if x in centro else ('arganzuela' if x in arganzuela else 
                                                                 ('retiro' if x in retiro else 
                                                                  ('salamanca' if x in salamanca else 
                                                                   ('chamartin' if x in chamartin else 
                                                                    ('tetuan' if x in tetuan else
                                                                     ('chamberi' if x in chamberi else 
                                                                      ('fuencarral' if x in fuencarral_pardo else 
                                                                       ('moncloa' if x in moncloa_arav else 
                                                                        ('latina' if x in latina else 
                                                                         ('carabanchel' if x in carabanchel else 
                                                                          ('usera' if x in usera else 
                                                                           ('p vallecas' if x in puente_vallecas else 
                                                                            ('moratalaz' if x in moratalaz else 
                                                                             ('ciudad_lineal' if x in ciudad_lineal else 
                                                                              ('hortaleza' if x in hortaleza else 
                                                                               ('villaverde' if x in villaverde else 
                                                                                ('v vallecas' if x in villa_vallecas else 
                                                                                 ('vicalvaro' if x in vicalvaro else 
                                                                                  ('barajas' if x in barajas else 'REVISAR')))))))))))))))))))
        data['zona'] = data['ubicacion'].apply(funcion_lambda)
        return data
    except Exception as a:
        print(f"No pude transformar el df por {a} en la función (zonas_nuevas)")

def obtener_coordenadas(data):
    '''Funcion para asignar coordenadas a las ubicaciones que tenemos en el dataframe'''
    try: 
        
        def obtener_coordenadas(direccion):
            try:
                geolocalizador = Nominatim(user_agent="maria")
                ubicacion = geolocalizador.geocode(direccion)

                if ubicacion:
                    latitud = ubicacion.latitude
                    longitud = ubicacion.longitude

                    return latitud, longitud
                else:
                    print(f"No se encontraron coordenadas para la dirección proporcionada {direccion}.")
                    time.sleep(1)
                    latitud = float(input('Lat: '))
                    longitud = float(input('Long: '))
                    return latitud, longitud
            except ValueError:
                pass

        dicc = []
        for i in data['ubicacion'].unique():
            coodenadas = obtener_coordenadas(i)
            dicc.append({'ubicacion' : i, 
                        'latitud' : coodenadas[0],
                        'longitud' : coodenadas[1]})
        df_coordenadas = pd.DataFrame(dicc)
    except Exception as a:
        print(f"No pude generar el df por {a}")
    return df_coordenadas

def grafico_variable_ppal(data):
    ''' 

    Funcion para graficar la variable principal a analizar
    input : df
    output: grafico

    '''
    try:
        media_color = 'r'
        mediana_color = 'b'
        media = data['precio_venta_por_m2'].mean()
        median = data['precio_venta_por_m2'].median()
        desv_std = data['precio_venta_por_m2'].std()  
        kurtosis_valor = kurtosis(data['precio_venta_por_m2'])
        simetria_valor = skew(data['precio_venta_por_m2'])

        sns.kdeplot(data=data, x='precio_venta_por_m2',fill=True,palette='hls')
        # Agregar líneas verticales para las estadísticas
        plt.axvline(media, color=media_color, linestyle='dashed', label=f'Media: {media:.2f}')
        plt.axvline(median, color= mediana_color, linestyle='dashed', label=f'Median: {median:.2f}')
        plt.axvline(desv_std, color='y', linestyle='dashed', label=f'Desv_std: {desv_std:.2f}')

        plt.title('Distribución del precio por metros cuadrados')
        # plt.xlabel('Popularidad')
        # plt.ylabel('Densidad')

        plt.legend()

        plt.show()
        # Interpretación de los valores

        print(f"kurtosis: {kurtosis_valor:.2f}")
        print(f"simetria: {simetria_valor:.2f}")

        if kurtosis_valor > 0:
            print("La distribución es leptocúrtica, lo que sugiere colas pesadas y picos agudos.")
        elif kurtosis_valor < 0:
            print("La distribución es platicúrtica, lo que sugiere colas ligeras y un pico achatado.")
        else:
            print("La distribución es mesocúrtica, similar a una distribución normal.")

        if simetria_valor < 0:
            print("La distribución es asimétrica positiva (sesgo hacia la derecha).")
        elif simetria_valor > 0:
            print("La distribución es asimétrica negativa (sesgo hacia la izquierda).")
        else:
            print("La distribución es perfectamente simétrica alrededor de su media.")
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def prueba_grafica_normalidad(data):
    '''Función para analizar graficamente si una variable tiene una distribución normal'''
    try: 
        probplot(data['precio_venta_por_m2'], dist="norm", plot=plt)
        plt.title('Q-Q Plot de Precio de venta por m2')
        plt.show()
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def prueba_normalidad_shapiro(data):
    '''Función para analizar si una variable tiene una distribución normal mediante shapiro'''
    try:
        stat, p_value = shapiro(data['precio_venta_por_m2'])
        print("Estadística de prueba:", stat)
        print("Valor p:", p_value)

        if p_value < 0.05:
            print("Rechazamos la hipótesis nula; los datos no siguen una distribución normal.")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula; los datos podrían seguir una distribución normal.")
    except Exception as a:
        print(f"No pude analizar la variable por {a}")

def pair_plot(data):
    '''Funcion para ver graficamente como se comportan algunas de  las variables cuantitativas'''
    try:
        df_cuant_pair_plot = data.select_dtypes(include = 'number').drop(columns=['annio_construccion', 'latitud', 'longitud'], axis=1)
        sns.pairplot(df_cuant_pair_plot, kind='reg', palette='husl', markers='.')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def precio_cee(data):
    '''Función para ver graficamente la relación que existe entre la letra del certificado energético y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_cee = data.groupby('cee', as_index=False).mean(numeric_only = True)
        sns.catplot(x = 'precio_venta_por_m2', y='cee', hue = 'cee', kind= 'bar',
        data=df_precio_venta_cee.sort_values(by='cee'), palette='husl')
        # ax.set_xticklabels(df_precio_venta_cee['cee'].sort_values().unique(), rotation=90)
        plt.title('Relación entre CEE y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def precio_tipo_inmueble(data):
    '''Función para ver graficamente la relación que existe entre el tipo de inmueble y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_tipo_inmueble = data.groupby('tipo_inmueble', as_index=False, sort=True).mean(numeric_only = True)
        sns.catplot(x= 'precio_venta_por_m2', y = 'tipo_inmueble', data = df_precio_venta_tipo_inmueble, kind='bar', hue = 'tipo_inmueble', palette='husl')
        # ax.set_xticklabels(df_precio_venta_tipo_inmueble['tipo_inmueble'].sort_values().unique(), rotation = -45)
        plt.title('Relación entre tipo de inmueble y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def tranformacion_numerica(data):
    '''Función para convertir los booleanos que tenemos en el dataframe en 1 y 0 para poder analizar otras cosas'''
    try:
        df_todo_n = data.copy()
        df_todo_n.replace(False, 0, inplace=True)
        df_todo_n.replace(True, 1, inplace=True)
    except Exception as a:
        print(f"No pude analizar la variable por {a}")
    return df_todo_n

def grafico_precio_var1_var2(data, var1, var2):
    '''Función para evaluar como aumenta el precio de venta por m2 respecto de las habitaciones y otra variable cualitativa a elección
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        sns.scatterplot(x= var1, y = 'precio_venta_por_m2', data = data, hue = var2, palette='husl')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_zona_yvariable(data, variable):
    '''Función para evaluar como aumenta el precio de venta por m2 por zona y otra variable a elegir
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    data.sort_values(by= 'precio_venta_por_m2')
    try:
        sns.catplot(x = 'precio_venta_por_m2', y='zona', hue = variable, kind= 'bar', palette='husl',
            data=data, errorbar = 'sd', err_kws={'linewidth': 1})
        # ax.set_xticklabels(data['zona'].sort_values().unique(), rotation = -45)
        plt.title(f'Relación entre {variable} y precio de venta por m2 por Zonas')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_tipo_inmueble_yvariable(data, variable):
    '''Función para evaluar como aumenta el precio de venta por m2 por tipo de inmueble y otra variable a elegir
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        sns.catplot(x = 'precio_venta_por_m2', y='tipo_inmueble', hue = variable, kind= 'bar', palette='husl',
            data=data, errorbar = 'sd', err_kws={'linewidth': 1})
        # ax.set_xticklabels(data['zona'].sort_values().unique(), rotation = -45)
        plt.title(f'Relación entre {variable} y precio de venta por m2 por Tipo de inmueble')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_zona(data):
    '''Función para evaluar como aumenta el precio de venta por m2 por zona y otra variable a elegir
    Input: 
    data = dataframe
    variable = columa dataframe
    '''
    try:
        orden = data.groupby('zona')['precio_venta_por_m2'].mean().sort_values(ascending=False).index
        sns.catplot(x = 'precio_venta_por_m2', y='zona', hue = 'zona', kind= 'bar', palette='husl',
            data=data, errorbar = 'sd', err_kws={'linewidth': 1}, order= orden)
        # ax.set_xticklabels(data['zona'].sort_values().unique(), rotation = -45)
        plt.title('Relación entre la zona y precio de venta por m2')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_precio_var1(data, variable):
    '''Función para ver graficamente la relación que existe entre la letra del certificado energético y el precio de venta por metro cuadrado'''
    try: 
        df_precio_venta_var = data.groupby(variable, as_index=False).mean(numeric_only = True)
        sns.catplot(y = 'precio_venta_por_m2', x=variable, hue = variable, kind= 'bar',
        data=df_precio_venta_var, palette= 'husl')
        # ax.set_xticklabels(df_precio_venta_cee[variable].sort_values().unique(), rotation=90)
        plt.title(f'Relación entre {variable} y Precio de venta por metros cuadrados')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_mapa(data, tipo):
    '''Función para graficar en un mapa las variables de precio venta por m2, por zona y según el tamaño que tienen'''
    try:
        if tipo == 't':
            fig = px.scatter_mapbox(data, lat = 'latitud', lon = 'longitud', size = 'metros_cuadrados',
                            color = 'precio_venta_por_m2', color_continuous_scale = 'plasma',
                            zoom = 3, mapbox_style = 'open-street-map')
            fig.show()
        elif tipo == 'z':
            fig = px.scatter_mapbox(data, lat = 'latitud', lon = 'longitud', size = 'precio_venta_por_m2',
                            color = 'zona', color_continuous_scale = 'plasma',
                            zoom = 3, mapbox_style = 'open-street-map')
            fig.show()
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_heatmap(data):
    '''Función para graficar en un mapa de calor mostrando las correlaciones entre las variables'''
    try:
        df_cuant = data.select_dtypes(include = 'number')
        plt.figure(figsize=(10,10))
        sns.heatmap(df_cuant.corr(numeric_only=True), robust=True, 
                    square = True, linewidths = .3, annot=True)      
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def grafico_var1_var2(data, var1, var2):
    '''Función para graficar la relacion de dos variables'''
    try:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=var1, y=var2, data=data, marker='o')

        plt.title(f'Relación entre {var1} y {var2}')
    except Exception as a:
        print(f"No pude hacer el gráfico por {a}")

def prueba_corr_spearman(df, var1, var2):
    try:
        correlation_coefficient, p_value = spearmanr(df[var1], df[var2])
        print(f"Coeficiente de correlación de Spearman: {correlation_coefficient}")
        print(f"Valor p: {p_value.round()}")

        alpha = 0.05
        if p_value < alpha:
            print("Hay evidencia para rechazar la hipótesis nula; existe una correlación significativa.")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula; no se puede afirmar una correlación significativa.")
    except Exception as a:
        print(f"No pude evaluar la correlación por {a}")

def prueba_krus_cee(data):
    try:
        # Prueba de Kruskal-Wallis para más de dos muestras independientes
        stat_kw, p_value_kw = kruskal(data['precio_venta_por_m2'][data['cee'] == 'A'],
                                    data['precio_venta_por_m2'][data['cee'] == 'B'],
                                    data['precio_venta_por_m2'][data['cee'] == 'C'],
                                    data['precio_venta_por_m2'][data['cee'] == 'D'],
                                    data['precio_venta_por_m2'][data['cee'] == 'E'],
                                    data['precio_venta_por_m2'][data['cee'] == 'F'],
                                    data['precio_venta_por_m2'][data['cee'] == 'G'],
                                    data['precio_venta_por_m2'][data['cee'] == 'inmueble exento'],
                                    data['precio_venta_por_m2'][data['cee'] == 'no indicado'],
                                    data['precio_venta_por_m2'][data['cee'] == 'en trámite']
                                    )
        alpha = 0.05 
        # Hipótesis nula (H0): No hay diferencia significativa en la de precios por m2 entre las letras de los certificados.
        # Hipótesis alternativa (Ha): Existe al menos una diferencia significativa en la de precios por m2 entre las letras de los certificados.

        print(f"\nPrueba de Kruskal-Wallis para más de dos muestras independientes: stat = {stat_kw}, p_value = {p_value_kw}")

        if p_value_kw < alpha:
            print("Rechazamos la hipótesis nula. Hay evidencia de al menos una diferencia significativa en la de precios por m2 entre las letras de los certificados")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula. No hay diferencia significativa en la de precios por m2 entre las letras de los certificados.")
    except Exception as a:
        print(f"No pude hacer la prueba {a}")

def prueba_krus_zonas(data):
    try:
    # Prueba de Kruskal-Wallis para más de dos muestras independientes
        stat_kw, p_value_kw = kruskal(data['precio_venta_por_m2'][data['zona'] == 'sur'],
                                    data['precio_venta_por_m2'][data['zona'] == 'norte'],
                                    data['precio_venta_por_m2'][data['zona'] == 'este'],
                                    data['precio_venta_por_m2'][data['zona'] == 'oeste'],
                                    data['precio_venta_por_m2'][data['zona'] == 'centro']
                                    )
        alpha = 0.05 
        # Hipótesis nula (H0): No hay diferencia significativa en en el precio por m2 entre las zonas.
        # Hipótesis alternativa (Ha): Existe al menos una diferencia significativa en en el precio por m2 entre las zonas.

        print(f"\nPrueba de Kruskal-Wallis para más de dos muestras independientes: stat = {stat_kw}, p_value = {p_value_kw}")

        if p_value_kw < alpha:
            print("Rechazamos la hipótesis nula. Hay evidencia de al menos una diferencia significativa en en el precio por m2 entre las zonas")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula. No hay diferencia significativa en en el precio por m2 entre las zonas.")
    except Exception as a:
        print(f"No pude hacer la prueba {a}")

def prueba_krus_tipo_inmueble(data):
    try:
    # Prueba de Kruskal-Wallis para más de dos muestras independientes
        stat_kw, p_value_kw = kruskal(data['precio_venta_por_m2'][data['tipo_inmueble'] == 'HouseType 1: Pisos'],
                                    data['precio_venta_por_m2'][data['tipo_inmueble'] == 'HouseType 5: Áticos'],
                                    data['precio_venta_por_m2'][data['tipo_inmueble'] == 'HouseType 2: Casa o chalet'],
                                    data['precio_venta_por_m2'][data['tipo_inmueble'] == 'HouseType 4: Dúplex']
                                    )
        alpha = 0.05 
        # Hipótesis nula (H0): No hay diferencia significativa en en el precio por m2 entre los tipos de inmuebles.
        # Hipótesis alternativa (Ha): Existe al menos una diferencia significativa en en el precio por m2 entre los tipos de inmuebles.

        print(f"\nPrueba de Kruskal-Wallis para más de dos muestras independientes: stat = {stat_kw}, p_value = {p_value_kw}")

        if p_value_kw < alpha:
            print("Rechazamos la hipótesis nula. Hay evidencia de al menos una diferencia significativa en en el precio por m2 entre los tipos de inmuebles")
        else:
            print("No hay suficiente evidencia para rechazar la hipótesis nula. No hay diferencia significativa en en el precio por m2 entre los tipos de inmuebles.")
    except Exception as a:
        print(f"No pude hacer la prueba {a}")

def transformar_cee(df):
    try:
        o_encoder = OrdinalEncoder(categories=[["no indicado", "inmueble exento", 
                                      "en trámite", "G", "F", "E", 
                                      "D", "C", "B", "A" 
                                      ]])
        o_encoder.fit(df[["cee"]])
        df["cee_e"] = o_encoder.transform(df[["cee"]])
        df.drop(columns= ['cee'], inplace= True)
        return df
    except Exception:
        print('No se pudo transformar la columna cee')
        traceback.print_exc()