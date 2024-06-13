import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
from huggingface_hub import login
login(token=HF_TOKEN)
with open("text.txt" , "r" , encoding="utf-8") as f:
    text = f.read()
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import PromptTemplate
llm = HuggingFaceInferenceAPI(model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' , token =HF_TOKEN)
template = (
    "We have provided context information below.\n"
    "----------------------\n"
    "{context_str}"
    "-----------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)
question = "What is the name of movie on serial number50"
prompt = qa_template.format(context_str=text , query_str = question)
response = llm.complete(prompt)
print(response.text)


