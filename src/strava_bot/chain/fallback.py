import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate, 
    MessagesPlaceholder, 
    ChatPromptTemplate
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
)

system_prompt = """

"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("Domanda utente da riscrivere: {input}")
])

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
    num_predict=1_000
)

rewrite_input_chain = (
    prompt
    | llm
)