import os
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-your-openai-key"


def get_menu(cuisine):
    llm = OpenAI()

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

    chain_seq = SequentialChain(chains=[restaurant_name_chain, menu_list_chain], input_variable=['cuisine'],
                                output_variable=['restaurant_name', 'menu_items'])
    return chain_seq({"cuisine": cuisine})