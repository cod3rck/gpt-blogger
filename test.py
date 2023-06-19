from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

# Load the API Key and endpoint from environment or replace with your actual API Key and endpoint
load_dotenv()
SHALE_API_KEY = os.getenv("SHALE_API_KEY")
os.environ['OPENAI_API_BASE'] = "https://shale.live/v1"
os.environ['OPENAI_API_KEY'] = SHALE_API_KEY  # use the variable here

# Initialize the model
llm = OpenAI(model='vicuna-13b-v1.1')

# Create a PromptTemplate
template = """Question: {question}

# Answer: Give the results based on existing knowledge."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Set your question
question = "How to run a successful start-up?"

# Run the LLMChain and print the result
result = llm_chain.run(question)
print(result)
