"""
RODAR UMA BUSCA PERSONALIZADA 
NO SERPER COM:
- locale
- n_results
- country
- location

USANDO O LLAMA 3.1-8b, gemma2 ou qwen2

"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os
import logging

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

arquivo_log = "execucao.log"
handler = logging.FileHandler(arquivo_log)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Llama3
grog_api_key = os.getenv("GROQ_API_KEY")
if not grog_api_key:
    raise ValueError(
        "A chave de API do GROG não foi fornecida. Verifique o arquivo .env."
    )
llama_3 = ChatGroq(
    api_key=grog_api_key,
    model="llama3-70b-8192",  # "llama-3.1-70b-versatile"
    timeout=180,
)

# Ollama
ollama = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434",
)
"""
os.environ["OPENAI_API_KEY"] = "NA"
ollama = ChatOpenAI(
    model="llama3.1",
    base_url="http://localhost:11434",
)"""

# Definindo LLM e rpm maximo padrao
DEFAULT_LLM = ollama
DEFAULT_MAX_RPM = None

# Ferramenta de pesquisa na internet
search_tool = SerperDevTool()
search_tool.country = "BR"
search_tool.location = "Rio de Janeiro"
search_tool.locale = "pt-BR"
search_tool.n_results = 25


pesquisador = Agent(
    role="Pesquisador de assunto",
    goal="Pesquisar informações detalhadas sobre {assunto} na internet",
    backstory=(
        "Você é um especialista em pesquisa, com habilidades aguçadas para "
        "encontrar informações valiosas e detalhadas na internet."
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=DEFAULT_LLM,
    allow_delegation=False,
)

tarefa_pesquisa = Task(
    description=(
        "Realizar uma pesquisa detalhada na internet para encontrar notícias atuais e em alta sobre o assunto: {assunto}. "
        "Foque em identificar artigos e reportagens recentes que estejam sendo amplamente discutidos ou que tenham grande relevância no momento. "
        "Busque por aspectos únicos, insights e dados relevantes que possam enriquecer a compreensão do assunto pelo usuário. "
        "A pesquisa deve ser organizada em um documento Markdown e todo o conteúdo deve estar em Português Brasil."
    ),
    expected_output=(
        "Um documento Markdown contendo as principais informações, dados e insights sobre {assunto}. "
        "O documento deve incluir links para as notícias encontradas e uma breve descrição de cada uma, destacando sua relevância."
    ),
    tools=[search_tool],
    output_file="pesquisa_noticias.md",
    agent=pesquisador,
    max_iter=2,
)


revisor = Agent(
    role="Revisor de Conteúdo",
    goal="Revisar todo o conteúdo produzido, incluir links das imagens geradas e entregar a versão final ao usuário",
    verbose=True,
    memory=True,
    backstory=(
        "Você tem um olho afiado para detalhes, garantindo que todo o conteúdo "
        "esteja perfeito antes de ser entregue ao usuário."
    ),
    llm=DEFAULT_LLM,
    allow_delegation=False,
)

tarefa_revisao = Task(
    description=(
        "Revisar e refinar todo o conteúdo produzido pelo agente 'Pesquisador de assunto' a partir da tarefa_pesquisa. "
        "O objetivo é criar uma compilação envolvente e bem estruturada das notícias encontradas, pronta para ser publicada em um blog de notícias. "
        "Incluir os links dos resultados das pesquisas de forma destacada e garantir que o conteúdo seja claro, fluido e agradável ao leitor. "
        "O texto deve estar em Português Brasil e deve ser formatado em Markdown, pronto para publicação."
    ),
    expected_output=(
        "Um documento em Markdown com o conteúdo revisado, resumido e organizado de maneira atraente. "
        "O documento deve conter seções claras, títulos e subtítulos apropriados, resumos envolventes das notícias, "
        "e links das fontes destacadas, criando um artigo de blog profissional e pronto para entrega ao usuário."
    ),
    agent=revisor,
    output_file="resumo_noticias.md",  # Configurando o output para salvar em um arquivo Markdown
)


# Formando a crew
crew = Crew(
    agents=[
        pesquisador,
        revisor,
    ],
    tasks=[
        tarefa_pesquisa,
        tarefa_revisao,
    ],
    process=Process.sequential,
    verbose=True,
    memory=True,
    logger=logger,
    manager_llm=DEFAULT_LLM,
    function_calling_llm=DEFAULT_LLM,
    max_rpm=DEFAULT_MAX_RPM,
    allow_delegation=False,
)

assunto = input(f"Quais notícias de {search_tool.location} gostaria de pesquisar? ")

print(assunto)

# Executando o processo com o assunto escolhido
result = crew.kickoff(inputs={"assunto": assunto})
logger.info(result)
print(f"Execução detalhada salva em {arquivo_log}")
