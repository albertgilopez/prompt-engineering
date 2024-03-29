# https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start
# !pip install langchain

import os 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

SYSTEM_PROMPT_TEMPLATE = '''
    Te voy a entregar un review de un cliente que compró la aspiradora {product_name}. \
    Del review extrae la siguiente información:

    gift: la compra fue un regalo para alguien o no? \
    Responde Si, si fue un regalo, No, si no fue un regalo, No se, si no se sabe \

    delivery_days: Cuantos días demoró el producto en llegar? \
        Si no encuentras esa información, devuelve un -1 \

    price_value: Extrae cualquier comentario referente al precio del producto,\
    Si no encuentras nada, devuelve un string vacío.

    Formatea la respuesta como JSON con las siguientes llaves:
    gift
    delivery_days
    price_value

    review: {review}
'''

CUSTOMER_REVIEW = '''
    La aspiradora salió muy buena, tiene tres ajustes: \
    alfombra gruesa, alfombra delgada y piso de madera. \
    Llegó en dos días, justo a tiempo para el cumpleaños de mi marido, a él le encanta limpiar la casa. \
    Es un poco más cara que las otras aspiradoras, \
    pero creo que vale la pena por las características adicionales.
'''

chat_template = ChatPromptTemplate.from_messages([
    ("system",SYSTEM_PROMPT_TEMPLATE),
])

messages = chat_template.format_messages(product_name="Powervac 2000", review=CUSTOMER_REVIEW)
response = chat.invoke(messages)

gift_schema = ResponseSchema(name="gift", description="Si la compra fue un regalo o no")
delivery_days_schema = ResponseSchema(name="delivery_days", description="Cuantos días demoró el producto en llegar")
price_value_schema = ResponseSchema(name="price_value", description="Cualquier comentario referente al precio del producto")

review_analysis_schema = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(review_analysis_schema)

output = output_parser.parse(response.content)
print(output)
print(type(output))

delivery_days = output.get('delivery_days')
gift = output.get('gift')
price_value = output.get('price_value')

print(delivery_days)
print(gift)
print(price_value)
