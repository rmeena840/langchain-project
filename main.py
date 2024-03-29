# this file is only for trying out the main components of langchain. Please check the real
# life project in the same project.
import os

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationChain
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain_core.tracers import langchain
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

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

# simple sequential chain
# create a simple chain that takes cuisine as input in first chain and gives good restaurant name
# and this restaurant name as input in second hain and gives menu

# first node of chain
restaurant_name_prompt = PromptTemplate.from_template("Suggest a good restaurant name for {cuisine} food")
restaurant_name_chain = LLMChain(llm=llm, prompt=restaurant_name_prompt)

# second node of chain
menu_list_prompt = PromptTemplate.from_template("Suggest menu for {restaurant}")
menu_list_chain = LLMChain(llm=llm, prompt=menu_list_prompt)

chain = SimpleSequentialChain(chains=[restaurant_name_chain, menu_list_chain])
response = chain.run("Indian")

# Sequential Chain
# first node of chain
restaurant_name_prompt_seq = PromptTemplate.from_template("Suggest a good restaurant name for {cuisine} food")
restaurant_name_chain_seq = LLMChain(llm=llm, prompt=restaurant_name_prompt, output_key="restaurant_name")

# second node of chain
menu_list_prompt_seq = PromptTemplate.from_template("Suggest menu for {restaurant}")
menu_list_chain_seq = LLMChain(llm=llm, prompt=menu_list_prompt, output_key="menu_items")

chain_seq = SequentialChain(chains=[restaurant_name_chain, menu_list_chain], input_variable=['cuisine'], output_variable=['restaurant_name', 'menu_items'])
chain_seq({"cuisine": "Indian"})

# agents
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

agent.run("What was the GDP of India in 2023?")

# memory
memory = ConversationBufferMemory()
memory_chain = LLMChain(llm=llm, prompt=menu_list_prompt_seq, memory=memory)

# conversation chain
convo = ConversationChain(llm=llm)
convo.run("Who is elon musk?")
convo.run("Who is bill gates?")
convo.run("Where will next FIFA will happen?")
convo.run("Who is the father of computer?")

# print the convo chai buffer
print(convo.memory.buffer)

# the cost is high in retaining the memory. please use ConversationWindowBufferMemory

