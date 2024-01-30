""" Ejemplo de extracción de datos de un documento PDF usando OpenAI y LangChain"""

from openai import OpenAI

import os
import json

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

pdf_file_path = "path/to/pdf/file"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

def resume_formatted_response(content):

    schema = """
    {
      "first_name": "",
      "last_name": "",
      "linkedin_url": "",
      "email_adress": "",
      "nationality": "",
      "skill": [""],
      "study": [
        {
          "start_date": "",
          "end_date": "",
          "description": ""
        }
      ],
      "studies": [
        {
          "degree": "",
          "university": "",
          "country": "",
          "grade": ""
        }
      ],
      "work_experience": [
        {
          "company": "",
          "job_title": ""
        }
      ],
      "hobby": [""]
    }
    """
    

    responses = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt = f"""Format the below response to the following JSON schema: {schema}

        Here's the content:

        {content}

        """,
        temperature=0.1,
        max_tokens=500
    )

    choices = [{"text": choice.text, "finish_reason": choice.finish_reason, "index": choice.index} for choice in responses.choices]

    response_usage = {
        "completion_tokens": responses.usage.completion_tokens,
        "prompt_tokens": responses.usage.prompt_tokens,
        "total_tokens": responses.usage.total_tokens
    }

    response_data = {
        "id": responses.id,
        "created": responses.created,
        "model": responses.model,
        "choices": choices,
        "usage": response_usage
    }

    formatted_response = []
    for choice in response_data["choices"]:
        if choice["finish_reason"] == "stop":
            formatted_response.append(choice["text"])

    return formatted_response

formatted_resume = resume_formatted_response(docs[0].page_content)

try:
    resume_data = json.loads(formatted_resume[0])

    print(json.dumps(resume_data, indent=3))

except json.JSONDecodeError as e:
    print("Error al decodificar la respuesta JSON:", e)
