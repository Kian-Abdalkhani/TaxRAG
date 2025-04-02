from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(
    model="llama3.2:3b"
)

template = """
You are an expert in answering questions related to the US tax code

Here is the US tax code: {tax_code}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model

while True:
    print("\n\n----------------------------------------------")
    question = input("What is your question? (enter q to quit):")
    print("\n\n")
    if question == "q":
        break

    tax_code = retriever.invoke(question)
    result = chain.invoke({"tax_code":tax_code,"question":question})
    print(result)
