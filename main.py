# Using Openai

import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)


# Using Gemini
# # News Research Tool
# import streamlit as st
# import asyncio
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# from langchain.chains import RetrievalQAWithSourcesChain
# import time
# import os

# # Ensure event loop exists for gRPC async calls
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)

# st.title("News Research Tool")
# st.sidebar.title("News Article URLs")

# # Collect URLs
# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}", key=f"url{i+1}")
#     urls.append(url)

# # Remove empty ones
# urls = [u for u in urls if u.strip()]

# process_url_clicked = st.sidebar.button("Process URLs")
# # file_path = "faiss_store.pkl"

# main_placeholder = st.empty()

# if process_url_clicked:
#     if not urls:
#         st.error("Please enter at least one valid URL.")
#         st.stop()

#     # Load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...")
#     data = loader.load()

#     # Split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000,
#     )
#     main_placeholder.text("Text Splitter ...Started...")
#     docs = text_splitter.split_documents(data)

#     # If no documents found
#     if not docs:
#         st.error("No content found from the provided URLs. Please check the links.")
#         st.stop()

#     # Create embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="text-embedding-004",
#         task_type="RETRIEVAL_DOCUMENT",
#     )

#     # Build FAISS vector store
#     main_placeholder.text("Embedding Vector Started Building...")
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     time.sleep(2)

#     # Save to file
#     vectorstore_openai.save_local("faiss_store")
#     st.success("Processing completed and FAISS store saved!")
#     # with open(file_path, 'wb') as f:
#     #     pickle.dump(vectorstore_openai, f)



# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists("faiss_store"):
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="text-embedding-004",
#             task_type="RETRIEVAL_DOCUMENT",
#         )
#         vectorstore = FAISS.load_local(
#             "faiss_store",
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

#         chain = RetrievalQAWithSourcesChain.from_llm(
#             llm=llm,
#             retriever=vectorstore.as_retriever()
#         )
#         result = chain({"question": query}, return_only_outputs=True)
        
#         st.header("Answer")
#         st.write(result["answer"])

#         # Display sources
#         sources = result.get("sources", "")
#         if sources:
#             st.subheader("Sources:")
#             for source in sources.split("\n"):
#                 st.write(source)
         



