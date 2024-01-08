# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import json

client = OpenAI() # Crear una instancia del cliente de OpenAI

def chat(message):

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=400,
        temperature=0.9,
        messages=[
            {
                "role": "system",
                "content": "Eres un robot asistente de muy mal humor, todas tus respuestas son negativas y sarcásticas."
            },
            {
                "role": "assistant",
                "content": "Hola, ¿en qué puedo ayudarte?"
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    print(completion)
    print(completion.choices[0].message)
    print(completion.choices[0].message.content)

    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    prompt_cost = (prompt_tokens * 0.03) / 1000
    completion_cost = (completion_tokens * 0.06) / 1000

    print("Coste del prompt: " + str(prompt_cost) + " USD")
    print("Coste del completion: " + str(completion_cost) + " USD")
    print("Coste total: " + str(prompt_cost + completion_cost) + " USD")
    print("1 millón de transacciones: " + str((prompt_cost + completion_cost) * 1000000) + " USD")

# Probamos su funcionamiento
chat("¿Cómo puedo hacer un pedido?")
