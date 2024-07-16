from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
import gradio as gr
from chromadb.utils import embedding_functions
import chromadb

hugging_face_token = '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'
login(token=hugging_face_token) # Use the login function from huggingface_hub


#DB
def create_db(docs):
    ##embeddings old
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=docs,persist_directory=persist_directory,
                    embedding=instructor_embeddings)


    ###new ver

    # # Instantiate chromadb instance. Data is stored on disk (a folder named 'my_vectordb' will be created in the same folder as this file).
    # chroma_client = chromadb.PersistentClient(path="my_vectordb")
    # # Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
    # # Select the embedding model to use.
    # # List of model names can be found here https://www.sbert.net/docs/pretrained_models.html
    # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    # vectordb= chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef)
    # # Add all the data to the vector database. ChromaDB automatically converts and stores the text as vector embeddings. This may take a few minutes.
    
    # # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
    # documents = []

    # # Store the corresponding menu item IDs in this array.
    # metadatas = []
    # # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
    # ids = []
    # id = 1

    # # Loop thru each line and populate the 3 arrays.
    # for i in range(len(docs)):
    #     if i==0:
    #         # Skip the first row (the column headers)
    #         continue

    #     documents.append(docs[i].page_content)
    #     metadatas.append({"item_id": docs[i].metadata['title']})
    #     ids.append(str(id))
    #     id+=1

    # vectordb.add(
    #     documents=documents,metadatas=metadatas,
    # ids=ids
    # )


    return vectordb

#Docs
def get_docs():
    loader = WebBaseLoader('https://www.thinkmentalhealthwa.com.au/supporting-others-mental-health/how-to-help/how-to-start-the-conversation/')
    documents = loader.load()
    
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitteddocs = text_splitter.split_documents(documents)
    return splitteddocs
def create_model(model_id):
    MAX_INPUT_TOKEN_LENGTH = 4096
    # Configurations for 4-bit quantization
    use_4bit = True
    device_map = {"": 0}  # Ensures model is loaded on the GPU
    device = 'cuda:0'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
    model_id,quantization_config=bnb_config,device_map=device_map
    #attn_implementation="flash_attention_2", # if you have an ampere GPU
)

    return model
#retriever chain
def create_chain(vectordb):
    model_id = "meta-llama/Llama-2-13b-chat-hf"
    model = create_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = ChatPromptTemplate.from_messages([
        ("system","answer following the context: {context}"),
        (MessagesPlaceholder(variable_name="chat_history")),
        ("human","{input}")

    ])
    chain = prompt|llm

    retriever = vectordb

    retriever_chain = create_retrieval_chain(
        retriever=retriever,chain=chain
    )


    return retriever_chain

def process_chat(chain,question,chat_history):
    response = chain.invoke(
        {
            "input":question,
            "chat_history":chat_history
        }
    )
    return response["answer"]

if __name__ == "__main__":
    doc = get_docs()
    vectordb = create_db(doc)
    chain = create_chain(vectordb)

    chat_history=[]

    while True:
        user_input = 'You : '

        response = process_chat(chain,user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("assistants:", response)

