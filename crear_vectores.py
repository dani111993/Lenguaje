import csv
import numpy as np

import extractores_caract as caract
import limpieza_txt as limp

# Función para leer un archivo CSV y retornar sus datos como una lista de un solo nivel
def leer_archivo(nombre_archivo):
    datos = []
    with open(nombre_archivo, newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            datos.extend(row)  # Agrega los elementos de la fila a la lista principal
    return datos

# Lee cada archivo CSV y almacena sus datos en una sola lista
sheinbaum = leer_archivo('datos/Sheinbaum.csv')
xochitl = leer_archivo('datos/Xochitl.csv')
maynez = leer_archivo('datos/Maynez.csv')

# Crea un solo vector con los datos de todos los candidatos
dataset=sheinbaum.copy()
dataset.extend(xochitl)
dataset.extend(maynez)


"""
#############################################
Ejemplo de uso para extraer características
#############################################
"""

corpus_vehiculos_electricos = [
    "me encanta mi nuevo vehículo #CarrosElectricos eléctrico es tan silencioso y eficiente nunca volveré a los vehículos de gasolina @Tesla",
    "este automóvil eléctrico es increíble la aceleración es suave y potente #CarrosElectricos y nunca más tendré que preocuparme por el costo de la gasolina",
    "estoy decepcionado con mi nuevo coche eléctrico la autonomía de la batería es mucho menor de lo que esperaba y cargarlo lleva una eternidad #NoElectricos",
    "no recomendaría este vehículo eléctrico #CarrosElectricos a nadie la calidad de construcción es deficiente y he tenido varios problemas con el sistema eléctrico",
    "mi experiencia con este automóvil eléctrico ha sido mixta si bien aprecio su impacto ambiental reducido todavía estoy preocupado por la infraestructura de carga y la autonomía limitada @Tesla"
]
corpus = limp.limpia(dataset)
print("banderita")
print(corpus)
v_caract_ejemplo=np.zeros((300,13))

for i in range(len(corpus)):
    print(f"ciclo:{i}")
    v_caract_ejemplo[i][0]=(caract.contar_palabras(corpus[i]))
    embeddig3d=caract.generar_embedding_3d(corpus[i])
    v_caract_ejemplo[i][1]=embeddig3d[0]
    v_caract_ejemplo[i][2]=embeddig3d[1]
    v_caract_ejemplo[i][3]=embeddig3d[2]
    v_caract_ejemplo[i][4]=(caract.tf_palabra_mas_comun(corpus[i]))
    v_caract_ejemplo[i][5]=(caract.idf_ultima_palabra(corpus[i],corpus))
    v_caract_ejemplo[i][6]=(caract.tfidf_palabra_medio(corpus[i],corpus))
    v_caract_ejemplo[i][7]=(caract.contar_arrobas(corpus[i]))
    v_caract_ejemplo[i][8]=(caract.contar_hash(corpus[i]))
    v_caract_ejemplo[i][9]=(caract.contar_verbos(corpus[i]))
    v_caract_ejemplo[i][10]=(caract.contar_adjetivos(corpus[i]))
    v_caract_ejemplo[i][11]=(caract.calcular_polaridad(corpus[i]))
    if i <= 99:
        v_caract_ejemplo[i][12]=1
    elif  99 < i <= 199:
        v_caract_ejemplo[i][12]=2
    else:
        v_caract_ejemplo[i][12]=3
        




"""Exportar el archivo CSV"""
# Ruta del archivo CSV de salida
ruta_archivo_csv = 'matriz_caracteristicas.csv'

# Exportar la matriz de características a un archivo CSV
with open(ruta_archivo_csv, 'w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    for fila in v_caract_ejemplo:
        escritor_csv.writerow(fila)

