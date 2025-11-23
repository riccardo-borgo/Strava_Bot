import logging

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import RunnableLambda
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
Sei un assistente con il compito di determinare se la domanda dell'utente è in tema oppure no ad un determinato argomento.
In particolare, devi verificare se la domanda è relativa all'allenamento fisico, ain particolare alla corsa.

# Istruzioni dettagliate:
- Analizza la domanda dell'utente.
- Comprendi se la domanda si attiene all'argomento dell'allenamento fisico e della corsa.
- Aiutati a rispondere con la storia della conversazione per contestualizzare meglio la domanda.
- Rispondi con {{"topic_check": 1}} se la domanda è relativa all'allenamento, {{"topic_check": 0}} altrimenti.

History:
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("Domanda utente: {input}")
])

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
    num_predict=512
)

class TopicCheck(BaseModel):
    topic_check: int = Field(..., description="1 se la domanda è relativa all'allenamento, 0 altrimenti")
    
    @field_validator("topic_check")
    def validate_topic(cls, value):
        if value not in (0, 1):
            raise ValueError("Il campo 'topic_check' deve essere 0 o 1")
        return value

def parse_topic_check(topic: TopicCheck) -> int:
    return topic.topic_check

topic_check_chain = (
    prompt
    | llm.with_structured_output(TopicCheck)
    | RunnableLambda(parse_topic_check)
)