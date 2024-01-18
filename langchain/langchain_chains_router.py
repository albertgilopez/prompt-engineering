# https://python.langchain.com/docs/modules/chains
# !pip install langchain

import os 
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router import MultiPromptChain

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llmOpenAI = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Se crea un prompt para cada asistente (sales, customer support y tech support)

sales_prompt = '''
    Eres Paula Comunicativa, Gerente de Relaciones Públicas del Banco Gana Dinero.
    Tienes un talento especial para las relaciones públicas y un profundo conocimiento
    de las estrategias de comunicación del banco. Eres la cara del banco en eventos,
    conferencias y en medios de comunicación, siempre representando la imagen y los 
    valores del banco con carisma y profesionalismo. Eres muy agradable.

    La consulta del cliente o la situación en la que te encuentras es la siguiente:
    {input}

'''

customer_support_prompt = '''
    Eres Luis Inversor, Asesor Financiero Personal en el Banco Gana Dinero.
    Tienes una amplia experiencia en asesoramiento financiero y una profunda comprensión
    de los productos de inversión que ofrece el banco. Tu enfoque es proporcionar a los 
    clientes asesoramiento personalizado para ayudarles a alcanzar sus objetivos financieros,
    escuchando atentamente sus necesidades y ofreciendo soluciones a medida. Eres serio y formal.

    La consulta del cliente es la siguiente:
    {input}

'''

tech_support_prompt = '''
    Eres Sofía Tecno, Especialista en Banca Digital del Banco Gana Dinero.
    Estás al día con las últimas tecnologías y tendencias en banca digital, y tu rol
    es ayudar a los clientes a navegar y aprovechar las herramientas digitales del banco.
    Eres paciente y pedagógica, siempre dispuesta a explicar de forma sencilla cómo 
    utilizar las aplicaciones y servicios online del banco para mejorar la experiencia 
    del cliente. Eres sarcástica y un poco fría.    

    La consulta del cliente es la siguiente:
    {input}

'''

# Se crea una lista de diccionarios con la información de cada prompt

prompt_infos = [
     {
        "name": "sales", 
        "description": "Asistente de ventas",
        "prompt_template": sales_prompt,
    },
    {
        "name": "customer_support", 
        "description": "Asistente de soporte al cliente respecto a inversiones",
        "prompt_template": customer_support_prompt,
    },
    {
        "name": "tech_support", 
        "description": "Asistente de soporte técnico respecto a la página web",
        "prompt_template": tech_support_prompt,
    },
]

# Se crea un diccionario con el nombre de cada prompt y su respectivo chain

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llmOpenAI, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print(f"Destinos disponibles:\n{destinations_str}")

# Si el input no es adecuado para ninguno de los destinos, se usa el default
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llmOpenAI, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """
    Given a raw text input to a 
    language model select the model prompt best suited for the input.
    You will be given the names of the available prompts and a
    description of what the prompt is best suited for.
    You may also revise the original input if you think that revising
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \\ name of the prompt to use or "DEFAULT"
        "next_inputs": string \\ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt 
    names specified below OR it can be "DEFAULT" if the input is not
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input 
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>
"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), 
)

router_chain = LLMRouterChain.from_llm(llmOpenAI, router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True
                        )

result = chain.invoke("Hola, tengo un problema con la web del banco")
print(result)