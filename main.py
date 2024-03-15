import os

from langchain_core.tracers import langchain
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = "sk-"

# two types of model

# LLM model

llm = OpenAI()
# single prompt
print(llm.invoke("hello"))
# multiple prompt
print(llm.generate(["this is prompt 1", "this is prompt 2"]).generations)

# Chat model

chat = ChatOpenAI()

print(chat.invoke([
    SystemMessage(content="You are funny teacher"),
    HumanMessage(content="What is 1+1?")
]))

# cache the prompt response. It works only when same prompt is passed.
langchain.cache = InMemoryCache()

# prompt
prompt = PromptTemplate.from_template("Tell me a company name that makes {product}")
print(prompt.format(product="pizza"))
LLMChain(llm=llm, prompt=prompt)

# chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are assistant that translates {input_lang} to {output_lang}"),
    ("human", "{text}")
])
chat_prompt.format_messages(input_lang="English", output_lang="Hindi", text="My name is Ravindra.")
LLMChain(chat=chat, chat_prompt=chat_prompt)
