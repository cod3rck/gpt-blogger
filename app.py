from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from dotenv import find_dotenv, load_dotenv
import requests
import json
import streamlit as st

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# serp request to get list of news

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': SERPAPI_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    print("search results: ", response_data)
    return response_data


# llm to choose the best articles

def find_best_article_urls(response_data, query):
    # turn json into string
    response_str = json.dumps(response_data)

    # create llm to choose best articles
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    You are a world class blogger & researcher, you are extremely good at find most relevant articles to certain topic;
    {response_str}
    Above is the list of search results for the query {query}.
    Please choose the best 3 articles from the list, return ONLY an array of the urls, do not include anything else; return ONLY an array of the urls
    """

    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template)

    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=True)

    urls = article_picker_chain.predict(response_str=response_str, query=query)

    # Convert string to list
    url_list = json.loads(urls)
    print(url_list)

    return url_list


# get content from each article & create a vector database

def get_content_from_urls(urls):   
    # use unstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data

def summarise(data, query):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(data)    

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create a blog post about {query}
    Please follow all of the following rules:
    1/ Make sure the content is engaging, informative with good data
    2/ The content should address the {query} topic very well
    3/ The content needs to be readable, well-structured, and easily understandable
    4/ The content needs to give audience actionable advice & insights too

    SUMMARY:
    """


    prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries

# Turn summarization into blog post
def generate_blog(summaries, query):
    summaries_str = str(summaries)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {summaries_str}

    You are a world class journalist & blogger, text above is some context about {query}
    Please write a comprehensive blog post about {query} using the text above, and following all rules below:
    1/ The post needs to be engaging, informative with good data
    2/ The post needs to address the {query} topic very well
    3/ The post needs to be written in a way that is easy to read and understand
    4/ The post needs to give audience actionable advice & insights too

    BLOG POST:
    """

    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    blog_post_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    blog_post = blog_post_chain.predict(summaries_str=summaries_str, query=query)

    return blog_post



def main():
    load_dotenv(find_dotenv())

    st.set_page_config(page_title="Autonomous researcher - Blog posts", page_icon=":memo:")

    st.header("Autonomous researcher - Blog posts :memo:")
    openaiapi = st.text_input("OpenAI API Key")
    query = st.text_input("Topic of blog post")

    openai.api_key = openaiapi

    if query:
        print(query)
        st.write("Generating blog post for: ", query)
        
        search_results = search(query)
        urls = find_best_article_urls(search_results, query)
        data = get_content_from_urls(urls)
        summaries = summarise(data, query)
        blog_post = generate_blog(summaries, query)

        with st.expander("search results"):
            st.info(search_results)
        with st.expander("best urls"):
            st.info(urls)
        with st.expander("data"):
            st.info(data)
        with st.expander("summaries"):
            st.info(summaries)
        with st.expander("blog post"):
            st.info(blog_post)

if __name__ == '__main__':
    main()
