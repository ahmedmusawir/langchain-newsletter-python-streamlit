import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

def pick_best_articles_urls(response_dict, query):
    """
    Use LLM to choose the best articles from search results and return their URLs.

    Args:
    response_dict (dict): The dictionary containing search results.
    query (str): The search query string.

    Returns:
    list: A list of URLs of the best articles.
    """
    # Convert dictionary to JSON string
    response_str = json.dumps(response_dict)

    # Initialize LLM
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0.7)

    # Define the prompt template
    template = """ 
      You are a world class journalist, researcher, tech, Software Engineer, Developer, and an online course creator.
      You are amazing at finding the most interesting, relevant, and useful articles on certain topics.
      
      QUERY RESPONSE: {response_str}
      
      Above is the list of search results for the query "{query}".
      
      Please choose the best 3 articles from the list and return ONLY an array of the URLs.  
      Do not include anything else -
      return ONLY an array of the URLs. 
      Also make sure the articles are recent and not too old.
      If the file, or URL is invalid, show www.google.com.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", template.format(response_str="{response_str}", query="{query}"))
        ]
    )

    # Combine the prompt and LLM into a sequence
    chain: Runnable = prompt_template | llm | StrOutputParser()

    # Create input for the chain
    input_data = {"response_str": response_str, "query": query}

    # Invoke the chain
    result = chain.invoke(input_data)

    # Convert string to list
    url_list = json.loads(result)
    return url_list
