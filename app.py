import streamlit as st

@st.cache_resource
def setup():
    # from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    # from langchain.vectorstores import Chroma
    from langchain_community.vectorstores import FAISS
    # from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    # from langchain.chains import ConversationalRetrievalChain
    # from langchain.memory import ConversationBufferMemory
    from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
    from langchain_google_genai import GoogleGenerativeAI
    import configparser
    import os
    BASE_PATH = ''
    config = configparser.ConfigParser()
    config.read(os.path.join(BASE_PATH, 'config.ini'))
    MODE = config.get('DEFAULT', 'MODE')
    GOOGLE_API_KEY = config.get('DEFAULT', 'GOOGLE_API_KEY')
    GOOGLE_MODEL = config.get('DEFAULT', 'GOOGLE_MODEL')
    NO_DOCUMENTS = int(config.get('DEFAULT', 'NO_DOCUMENTS'))
    CHUNK_SIZE = int(config.get('DEFAULT', 'CHUNK_SIZE'))
    REASON = int(config.get('DEFAULT', 'REASON'))
    BOOK = config.get('DEFAULT', 'BOOK')
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print(f"Mode: {MODE}")
    print(f"Google API Key: {GOOGLE_API_KEY}")
    print(f"Google Model: {GOOGLE_MODEL}")
    print(f"Number of retrieved documents: {NO_DOCUMENTS}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Explain reasoning: {REASON}")
    print(f"Book File: {BOOK}")

    try:
        loader = PyPDFLoader(os.path.join(BASE_PATH, BOOK))
    except:
        try:
            loader = TextLoader(os.path.join(BASE_PATH, BOOK))
        except:
            print('Could not read the book.')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    llm = GoogleGenerativeAI(model=GOOGLE_MODEL)
    embeddings = HuggingFaceEmbeddings()
    if MODE == 'rag':
      docsearch = FAISS.from_documents(texts, embeddings) 
    else:
      text_contents = [doc.page_content for doc in texts]
      text_embeddings = embeddings.embed_documents(text_contents)  # Embed the text contents
      text_embedding_pairs = zip(text_contents, text_embeddings) 
      hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(llm, embeddings, "web_search")
      docsearch = FAISS.from_embeddings(text_embedding_pairs, hyde_embeddings)

    retriever = docsearch.as_retriever(search_kwargs={"k": NO_DOCUMENTS})

    if REASON == 1:
        string = " and explain your reasoning for your response."
    else:
        string = "."
    prompt_template = "You are a chatbot that uses context from a book to respond to users in English language. If the context is not relevant to OR not answer the user's query, politely decline to respond and redirect the user to the topic of the book. Also, do not format text in your response" + string
    prompt_template += """
[CONTEXT]
{context}
[QUERY] 
{question}
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = GoogleGenerativeAI(model=GOOGLE_MODEL)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True)
    return qa



st.title("Chat with your book")
query = st.text_area("Your query:")
if st.button("Generate"):
    qa = setup()
    if query:
        with st.spinner("Generating response..."):
            response = qa.invoke(query)
            try:
                reference = f"\nReference pages: {[doc.metadata['page'] for doc in response['source_documents']]}"
            except:
                reference = ""
            st.text_area("Response: ", value = response['result'] + reference, height = 200)
    else:
        st.warning("Please enter a question.")