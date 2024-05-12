import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
#from spacy.cli.download import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('cess_esp')
# nltk.download('punkt')
# nltk.download('opinion_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# download("es_core_news_sm")
# download("es_core_news_md")


"""Función para contar palabras"""
def contar_palabras(cadena_texto):
    palabras = cadena_texto.split()  # Divide la cadena de texto en palabras
    return len(palabras)


"""Función para obtener un embedding de 3 elementos dada una cadena de texto como entrada """
def generar_embedding_3d(texto):
    # Cargar el modelo descargado
    nlp = spacy.load("es_core_news_md")

    # Procesar el texto
    doc = nlp(texto)

    # Obtener el vector de características promedio para el texto
    embedding = doc.vector[:3]  # Tomar las primeras 3 dimensiones del vector

    return embedding


"""Función para obtener tf de la palabra más frecuente en el documento."""
def tf_palabra_mas_comun(texto):
    # Tokenizar el texto en palabras
    palabras = word_tokenize(texto)

    # Calcular la frecuencia de cada palabra
    frecuencia_palabras = Counter(palabras)

    # Encontrar la palabra más común y su frecuencia
    palabra_mas_comun, frecuencia_mas_comun = frecuencia_palabras.most_common(1)[0]

    return frecuencia_mas_comun


"""Función para obtener IDF de la última palabra de la cadena"""
def idf_ultima_palabra(texto, corpus):
    # Tokenizar el texto en palabras
    palabras_texto = word_tokenize(texto)

    # Obtener la última palabra no precedida por "#" o "@"
    ultima_palabra = None
    for palabra in reversed(palabras_texto):
        if not (palabra.startswith("#") or palabra.startswith("@")):
            ultima_palabra = palabra
            break

    if ultima_palabra is None:
        return None  # No se encontró ninguna palabra adecuada para calcular IDF

    # Contar la frecuencia de documentos que contienen la última palabra
    documentos_con_palabra = sum(1 for doc in corpus if ultima_palabra in word_tokenize(doc))

    # Calcular IDF
    idf = math.log(len(corpus) / (documentos_con_palabra + 1))  # Se suma 1 para evitar división por cero

    return idf


"""Función para obtener TF-IDF de la palabra que se encuentra al medio de la cadena"""
def tfidf_palabra_medio(texto, corpus):
    # Tokenizar el texto en palabras
    palabras_texto = word_tokenize(texto)

    # Encontrar la palabra en el medio del texto
    palabra_medio = None
    if len(palabras_texto) % 2 == 0:  # Si hay un número par de palabras
        indice_medio = len(palabras_texto) // 2
        palabra_medio = palabras_texto[indice_medio - 1]  # Tomar la palabra anterior al verdadero medio
    else:  # Si hay un número impar de palabras
        indice_medio = len(palabras_texto) // 2
        palabra_medio = palabras_texto[indice_medio]

    # Si la palabra en el medio comienza con "#" o "@", buscar una palabra antes de esta
    while palabra_medio.startswith("#") or palabra_medio.startswith("@"):
        indice_medio -= 1
        if indice_medio < 0:
            return None  # No se encontró ninguna palabra adecuada para calcular TF-IDF
        palabra_medio = palabras_texto[indice_medio]

    # Contar la frecuencia de la palabra en el corpus
    frecuencia_palabra = sum(1 for doc in corpus if palabra_medio in word_tokenize(doc))

    # Calcular IDF
    idf = math.log(len(corpus) / (frecuencia_palabra + 1))  # Se suma 1 para evitar división por cero

    # Calcular TF (Term Frequency)
    tf = palabras_texto.count(palabra_medio) / len(palabras_texto)

    # Calcular TF-IDF
    tf_idf = tf * idf

    return tf_idf


"""Función para contar la cantidad de símbolos '@' en una cadena de texto."""
def contar_arrobas(cadena_texto):
    return cadena_texto.count('@')


"""Función para contar la cantidad de símbolos '#' en una cadena de texto."""
def contar_hash(cadena_texto):
    return cadena_texto.count('#')


"""Función para obtener el conteo de verbos en español"""
def contar_verbos(texto):

    # Cargar el modelo en español de Spacy
    nlp = spacy.load("es_core_news_sm")

    # Procesar el texto
    doc = nlp(texto)

    # Filtrar verbos
    verbos = [token.text for token in doc if token.pos_ == 'VERB']

    return len(verbos)


"""Función para obtener el conteo de adjetivos en español"""
def contar_adjetivos(texto):

    # Cargar el modelo en español de Spacy
    nlp = spacy.load("es_core_news_sm")

    # Procesar el texto
    doc = nlp(texto)

    # Filtrar verbos
    adjetivos = [token.text for token in doc if token.pos_ == 'ADJ']

    return len(adjetivos)

"""Función para obtener el análisis de sentimiento de una cadena de texto"""
sid = SentimentIntensityAnalyzer()
def calcular_polaridad(texto):
    # Obtener puntajes de sentimiento
    scores = sid.polarity_scores(texto)
    
    # Clasificar el sentimiento basado en el puntaje
    if scores['compound'] >= 0.05:
        return 1  # Positivo
    elif scores['compound'] <= -0.05:
        return -1  # Negativo
    else:
        return 0  # Neutro





