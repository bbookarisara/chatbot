from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# def create_db():

#     client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
#                                     persist_directory="db"
#                                 ))
    
#     collection = client.create_collection(name="Chat_history")
    

#     loader = WebBaseLoader('https://www.thinkmentalhealthwa.com.au/supporting-others-mental-health/how-to-help/how-to-start-the-conversation/')
#     documents = loader.load()
    
#     #splitting the text into
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splitteddocs = text_splitter.split_documents(documents)
#     ##embeddings old
#     instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     persist_directory = 'db'
#     vectordb = Chroma.from_documents(documents=docs,persist_directory=persist_directory,
#                     embedding=instructor_embeddings)

#     return vectordb

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory="db"
                                ))
    
collection = client.create_collection(name="Chat_history")
student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world."""

collection.add(
    documents = [student_info, club_info, university_info],
    metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
    ids = ["id1", "id2", "id3"]
)