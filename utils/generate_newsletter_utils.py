from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


def generate_newsletter(summries, query):
    # converting summary into string
    summaries_str = str(summries)

    # creating the model
    llm = ChatOpenAI(model="gpt-4o", 
                     temperature=.7)
    # prompt engineering
    template = """
    {summaries_str}
        As a world class journalist, researcher, article, newsletter and blog writer, 
        you'll use the text above as the context about {query}
        to write an excellent newsletter to be sent to subscribers about {query}.
        
        This newsletter will be sent as an email.  The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Make sure to write it informally - no "Dear" or any other formalities.  Start the newsletter with
        `Hi All!
          Here is your weekly dose of the [CHANGE THIS ACCORDING TO {query}]  Newsletter, a list of what I find interesting
          and worth and exploring.`
          
        Make sure to also write a backstory about the topic - make it personal, engaging and lighthearted before
        going into the meat of the newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the conent is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actinable advice & insights including resouces and links if necessary.
        
        If there are books, or products involved, make sure to add amazon links to the products or just a link placeholder.
        
        As a signoff, write a clever quote (FORMAT IN MARKDOWN) related to learning, general wisdom, living a good life.  Be creative with this one - and then,
        Sign with (replace the "Sign with" with "Sincerely,")
        "The Moose 
          - The AI Guru"
        
          
        NEWSLETTER (GENERATE THE OUTPUT IN MARKDOWN FORMAT (but don't use markdown at the beginning) with proper headers like "## for h2, > for quotes etc.")-->:
    """

    # Prepping the Prompt Template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", template.format(summaries_str="{summaries_str}", query="{query}"))
        ]
    )

    # Combine the prompt and LLM into a sequence
    newsletter_chain: Runnable = prompt_template | llm | StrOutputParser()

    # Create input for the chain
    input_data = {"summaries_str": summaries_str, "query": query}

    # Invoke the chain
    result = newsletter_chain.invoke(input_data)

    # display the news letter
    # print('=====================================================================')
    # print(result)
    return result
