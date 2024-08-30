"""
RODAR UMA BUSCA PERSONALIZADA 
NO SERPER COM:
- locale
- n_results
- country
- location

USANDO O LLAMA 3.1-8b, gemma2 ou qwen2

"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Ollama
"""
os.environ["OPENAI_API_KEY"] = "NA"
ollama = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434",
)"""
ollama = ChatOpenAI(
    model="llama3.1",
    base_url="http://localhost:11434",
)
