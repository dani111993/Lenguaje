import nltk
import re
import unicodedata
from unicodedata import normalize
import inflect
from num2words import num2words
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwordsESP = set(stopwords.words('spanish'))

#quitar stopwords   (considera los stopwords de esa lista si tienen)(tampoco hace limpieza de texto)
def stopGuords(texto):
    palabras = texto.split()
    sinStop = [palabra for palabra in palabras if palabra.lower() not in stopwordsESP]
    return ' '.join(sinStop)

#QUITAR ACENTOS
def acentos(texto):
    return re.sub( r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",
        normalize( "NFD", texto), 0, re.I)

#Quitar signos de puntuacion  (Cualquiera de los tres sirve, no me decidi jajaja)(pero recomiendo que usen la de puntuacion2)
def puntuacion(texto):
    return re.sub(r'[^\w\s\d_]', '', texto)
def puntuacion2(texto):
    return re.sub(r'[^a-zA-Z0-9\s#@]', '', texto)
def signos(textt):
    return ''.join((c for c in unicodedata.normalize('NFD', textt) if unicodedata.category(c) != 'Mn'))


#convertir a Numeros a letra
def numeros(text):
    patron = r'\b\d+\b'
    def reemplazar(match):
        numero = int(match.group(0))
        return num2words(numero, lang='esp')
    numConvertido = re.sub(patron, reemplazar,text)
    return numConvertido

#Transforma a minusculas
def minusculas(texto):
    resultado = ""
    for caracter in texto:
        if caracter.isupper():
            resultado += caracter.lower()
        else:
            resultado += caracter
    return resultado

def limpia(texto):
    return [minusculas(numeros(puntuacion2(acentos(tuit)))) for tuit in texto]