import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Any

def run_llm(query: str, chat_history: List[Dict[str, Any]]):
  embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    base_url="http://localhost:1234/v1",
    api_key="key",
    model="text-embedding-nomic-embed-text-v1.5"
  )
  docsearch = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
  chat = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="key",
    model="deepseek-r1-distill-qwen-7b",
    verbose=True,
    temperature=0
  )

  retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
  stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
  rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
  history_aware_retriever = create_history_aware_retriever(
    llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
  )
  qa = create_retrieval_chain(
    retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
  )
  result = qa.invoke(input={"input": query, "chat_history": chat_history})

  new_result = {
    "query": query,
    "result": result["answer"],
    "source_document": result["context"]
  }
  return new_result


if __name__ == "__main__":
  res = run_llm(query="What is a LangChain Chain?")
  print(res)