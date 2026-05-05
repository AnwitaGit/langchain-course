from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai (pronounced: /ˈsʊndɜːr pɪˈtʃeɪ/), is an Indian–American business executive who has been the CEO of Google since 2015 and the CEO of its parent company Alphabet Inc. since 2019.[4][5][6]

Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[7] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

As of February 2026, his net worth is estimated at US$1.6 billion.[8]
    """

    summary_template = """Given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts abt them 
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template = summary_template
    )
    #llm = ChatOpenAI(temperature=0, model = "gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke( {"information": information})
    print(response.content)
if __name__ == "__main__":
    main()
