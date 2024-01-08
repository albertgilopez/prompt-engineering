# Habla con videos de YouTube

import re
import uuid
import json
from pytube import YouTube
import whisper
import chromadb
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os

client = OpenAI()  # Crear una instancia del cliente de OpenAI

CHROMA_DB_PATH = "path_to_chroma_db"
QUERY = "Your query here"

def parse_segment(segment):
    return {
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    }

print("\nDESCARGANDO VÍDEO...\n")

# URL del video de YouTube
ytvideo = YouTube("YouTube video URL here")
file_name = re.sub(r'\W+', '', ytvideo.title) + ".mp4"

# Define la ruta donde quieres guardar el video
output_path = "path_to_output"

# Comprueba si el video ya está descargado
if os.path.exists(os.path.join(output_path, file_name)):
    print("El video ya está descargado.")
else:
    ytvideo.streams.first().download(output_path, file_name)
    print("El video ha sido descargado.")

print("\nTRANSCRIBIENDO VÍDEO...\n")

transcription_file = "path_to_transcription_file"

# Comprobar si el archivo está creado
if os.path.exists(transcription_file):
    print("El archivo de transcripción se ha creado correctamente.")
else:
    # Transcribe el video
  
    print(f"Transcribiendo el archivo: {file_name}")
    print("Este proceso puede tardar unos minutos... Un momento, por favor.\n")

    MODEL = whisper.load_model("tiny.en")
    transcription = MODEL.transcribe("path_to_your_video")

    # Guardar la transcripción en un archivo de texto
    transcription_file = "path_to_output"
    with open(transcription_file, "w") as file:
        file.write(transcription["text"])

    segments = []
    for item in transcription["segments"]:
        segments.append(parse_segment(item))

    # Vectorizamos la transcripción y la guardamos en Chroma DB

    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    print("\nCREANDO EMBEDDINGS...\n")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    collection = chroma_client.create_collection("transcription")

    for segment in segments:
        
        print(f"Adding segment {segment['start']} - {segment['end']}")
        collection.add(
            ids=str(uuid.uuid4()),
            documents=segment["text"],
            metadatas={"start": segment["start"],
                    "end": segment["end"]})

        collection = chroma_client.get_collection(name="transcription")
    else:
        print("La base de datos Chroma DB no existe.")
      
print("\nCONSULTANDO EMBEDDINGS...\n")

# Código para consultar embeddings
if os.path.exists(CHROMA_DB_PATH):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name="transcription")
else:
    print("La base de datos Chroma DB no existe.")

if collection:
    result = collection.query(query_texts=QUERY, n_results=5)

    json_result = json.dumps(result, indent=4)
    print(json_result)

    # Ask question to bot
    SEGMENTS_TEXT = "\n".join([document for document in result['documents'][0]])

    SYSTEM_PROMPT = f"""
        Eres un bot especializado en responder preguntas acerca de videos de youtube.
        A continuación te voy a entregar parte de la transcripción de un video titulado '{ytvideo.title}',que se trata de '{ytvideo.description}',
        y tu vas a tener que contestar una pregunta del usuario, sólo basandote en la información que te estoy entregando. La información es la siguiente: {SEGMENTS_TEXT}"""
    
    print(SYSTEM_PROMPT)

else:
    print("No se puede realizar la consulta porque la base de datos Chroma DB no existe.")

print("\nCONECTANDO CON OPENAI...\n")

# Código para conectar con OpenAI
result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role":"user", "content": QUERY}
    ]
)

# print(result)
print(result.choices[0].message.content)

# CALCULAR COSTE DE TOKENS
token_count = len(result.choices[0].message.content.split())
cost_per_token = 0.0015  # Coste por token en dólares de GPT-3.5-TURBO

total_cost = token_count * cost_per_token
print(f"El coste total de los tokens es: ${total_cost}")
