from openai import OpenAI

import pandas as pd 
import numpy as np 

# from openai.embeddings_utils import get_embedding, cosine_similarity #deprecated

# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

client = OpenAI() # Crear una instancia del cliente de OpenAI


# Vamos a comparar la similtud entre estas frases. Utilizaremos la métrica de similitud de coseno con embeddings de texto 

# La métrica de similitud de coseno con embeddings de texto se basa en la representación vectorial de las frases utilizando modelos de embeddings de texto. 
# Los embeddings de texto capturan la semántica y el significado de las palabras y las frases en un espacio vectorial de alta dimensionalidad.
# La similitud de coseno es una medida que compara la similitud direccional entre dos vectores en un espacio vectorial. En el contexto de embeddings de texto, se utiliza la similitud de coseno para calcular la similitud entre los vectores de embeddings de dos frases.

# El cálculo de la similitud de coseno se realiza utilizando la fórmula:

# similitud = cos(θ) = (A · B) / (||A|| * ||B||)

# Donde A y B son los vectores de embeddings de las frases y (A · B) es el producto escalar entre los dos vectores. ||A|| y ||B|| representan las normas (longitudes) de los vectores A y B, respectivamente.

# La similitud de coseno devuelve un valor entre -1 y 1, donde 1 indica una similitud perfecta, 0 indica ninguna similitud y -1 indica una similitud inversa.
# Al calcular la similitud de coseno entre los embeddings de dos frases, se evalúa la similitud direccional entre los vectores. Si los vectores tienen una dirección similar, es decir, están más cerca en el espacio vectorial, la similitud de coseno será mayor, lo que indica una mayor similitud en el contenido semántico de las frases.

# Utilizando esta métrica, podemos comparar la similitud entre múltiples frases y determinar qué tan similares son en términos de contenido y significado.

# Algunos casos de uso en los que se puede aplicar:

# - Búsqueda de similitud de documentos: Puedes utilizar embeddings de texto y la similitud de coseno para buscar documentos similares en un conjunto de documentos. Esto es útil en motores de búsqueda de documentos, recomendación de contenido similar o agrupación de documentos relacionados.
# - Clasificación de texto: Puedes utilizar la similitud de coseno entre los embeddings de texto y representaciones de clase para clasificar texto en categorías o temas específicos. Esto puede ser útil en tareas como análisis de sentimiento, detección de spam o categorización de noticias.
# - Búsqueda semántica: Puedes utilizar embeddings de texto y la similitud de coseno para buscar palabras o frases semánticamente similares a una consulta dada. Esto es útil en completado automático de búsqueda, sugerencia de consultas de búsqueda o generación de sinónimos.
# - Agrupación de texto: Puedes utilizar la similitud de coseno para agrupar texto en grupos basados en similitud semántica. Esto puede ser útil en la segmentación de clientes, análisis de comentarios de usuarios o análisis de temas en redes sociales.

# - Comparación de similitud de oraciones: Puedes utilizar la similitud de coseno para comparar la similitud entre oraciones o frases. Esto es útil en tareas como detección de plagio, resumen de texto, respuesta a preguntas basada en contexto o corrección gramatical.

# En este último caso de uso, podríamos crear una fórmula básica para detectar plagio en los examenes:

# PREPARACIÓN DE LOS DATOS
# - Tener un conjunto de respuestas de los exámenes que se utilizarán para la comparación.
# - Convertir cada respuesta en un vector de embeddings utilizando un modelo de embeddings de texto, como BERT, GloVe o FastText. Esto se puede hacer utilizando bibliotecas como sentence-transformers o gensim.

# CÁCLULO DE SIMILIUDES
# - Calcular la similitud de coseno entre todas las combinaciones de respuestas utilizando los vectores de embeddings.
# - Cuanto mayor sea el valor de similitud de coseno, mayor será la similitud entre las respuestas.

# ESTABLECER UN UMBRAL DE SIMILITUD
# - Determinar un umbral de similitud por encima del cual considerarás que las respuestas están plagadas. El umbral dependerá de la sensibilidad y el contexto de tu aplicación.

# IDENTIFICAR SIMILITUDES SOSPECHOSAS
# - Identificar las combinaciones de respuestas que superen el umbral de similitud establecido. Estas combinaciones indicarán las respuestas que podrían tener contenido similar o copiado.

# Es importante tener en cuenta que este enfoque proporciona una indicación de posibles similitudes y requiere una revisión humana adicional para confirmar si se trata realmente de plagio.
# Además, la efectividad de esta técnica puede depender de la calidad y cantidad de datos de entrenamiento disponibles y del modelo de embeddings utilizado.

# Para una detección de plagio más precisa hay que utilizar enfoques más sofisticados y avanzados, como técnicas de procesamiento de lenguaje natural y algoritmos de comparación más complejos, para obtener resultados más precisos.


resp = client.embeddings.create(
		input = ["The maratton runner sprints towrds the finish line",
		"The student study diligently for examns",
		"The cyclist pedals furiously to complete the race",
		"The researchers analyze data meticulously"],
		model = "text-embedding-ada-002")

print(resp)
# print(resp.data[0].embedding)

embedding_a = resp.data[0].embedding
embedding_b = resp.data[1].embedding
embedding_c = resp.data[2].embedding
embedding_d = resp.data[3].embedding

# Utilizando el paquete NumPy

similaritya_b = np.dot(embedding_a, embedding_b)
print("La similitu entre la frase A y B es: {}.".format(similaritya_b))

similaritya_c = np.dot(embedding_a, embedding_c)
print("La similitu entre la frase A y C es: {}.".format(similaritya_c)) # hay más similitud porque las dos frases hablan de deporte


# Vamos a comparar la similtud entre estas otras frases

resp = client.embeddings.create(
		input = ["The chef prepares a mouthwatering dish",
		"The sun sets behind the mountains",
		"The pastry is so delicious",
		"The rainbows paint the sky beautifully"],
		model = "text-embedding-ada-002")

embedding_a = resp.data[0].embedding
embedding_b = resp.data[1].embedding
embedding_c = resp.data[2].embedding
embedding_d = resp.data[3].embedding

# Utilizando el paquete NumPy

similaritya_b = np.dot(embedding_a, embedding_b)
print("La similitu entre la frase A y B es: {}.".format(similaritya_b))

similaritya_c = np.dot(embedding_a, embedding_c)
print("La similitu entre la frase A y C es: {}.".format(similaritya_c)) # hay más similitud porque las dos frases hablan de comida

similarityb_d = np.dot(embedding_b, embedding_d)
print("La similitu entre la frase B y D es: {}.".format(similarityb_d)) # hay más similitud porque las dos frases hablan de la naturaleza

