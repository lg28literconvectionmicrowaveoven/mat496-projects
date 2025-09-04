# TODO: set up and use RedisVL as vector store
from dotenv import load_dotenv
from IPython.display import Markdown
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("lotr.txt", "r") as file:
    text = file.read()

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = RedisVectorStore(embeddings)
splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
search_results = vector_store.similarity_search_with_score(
    "Who does Frodo trust with the ring?",
    k = 10
)
doc_content = "\n".join(doc[0].page_content for doc in search_results)

prompt_template = """
You are an assistant that reads literary works and provides accurate and irrefutable facts about them or in them as stated. Use the following pieces of text from the literary work under scrutiny as context and answer the question to the best of your ability. If you don't know the answer, just say you don't know. Do not try to make up an answer.
Question: {question}
Context: {context}
"""
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

response = model.invoke(
    prompt_template.format(
        context=doc_content,
        question="Who does Frodo trust the most with the One Ring?"
    )
)

print(Markdown(response.content))
