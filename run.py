from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from peft import AutoPeftModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
import gradio as gr


hugging_face_token = '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'
login(token=hugging_face_token) # Use the login function from huggingface_hub

# response=llm.invoke("hello what is my name")
#print(response)


chat_history = []

#DB
def create_db(docs):
    #embeddings
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=docs,persist_directory=persist_directory,
                    embedding=instructor_embeddings)
    return vectordb

#Docs
def get_docs():
    loader = WebBaseLoader('https://www.thinkmentalhealthwa.com.au/supporting-others-mental-health/how-to-help/how-to-start-the-conversation/')
    documents = loader.load()
    
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitteddocs = text_splitter.split_documents(documents)
    return splitteddocs

#retriever chain
def create_chain(vectordb):
    model_id = "bbookarisara/llama-3-8b-japanese-arisara"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    #attn_implementation="flash_attention_2", # if you have an ampere GPU
)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = ChatPromptTemplate.from_messages([
        ("system","answer following the context: {context}"),
        (MessagesPlaceholder(variable_name="chat_history"))
        ("human","{message}")

    ])
    chain = prompt|llm

    retriever = vectordb.as_retriever()

    retriever_chain = create_retrieval_chain(
        retriever=retriever,chain=chain
    )


    return retriever_chain






# Function to generate answer
def generate_answer(message,history):
    docs = get_docs()
    vectordb=create_db(docs)
    chain = create_chain(vectordb)
    response = chain.invoke({
        "input": message,
        "context":"mental health"})
    #chat_history
    # chat_history.append()

    return response["answer"]

answer_bot = gr.ChatInterface(
                            generate_answer,
                            chatbot=gr.Chatbot(height=300),
                            textbox=gr.Textbox(placeholder="Ask me a question about nutrition and health", container=False, scale=7),
                            title="Mental Health ChatBot",
                            theme="soft",
                            cache_examples=False,
                            retry_btn=None,
                            undo_btn=None,
                            clear_btn=None,
                            submit_btn="Ask",
                            stop_btn="Interrupt",
                        )


if __name__ == "__main__":
    answer_bot.launch()
