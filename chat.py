import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
os.environ["api_key"] = st.secrets["secrets"]["api_key"]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ["api_key"])

st.set_page_config(
    page_title="ChatCUD",
    page_icon="💬",
    )


page_config = {
    st.markdown(
    "<h1 style='text-align: center; color: #b22222; font-family: Arial, sans-serif; background-color: #292f4598;'>chatCUD 💬</h1>",
    unsafe_allow_html=True
    ),
    st.markdown("<h4 style='text-align: center; color: white; font-size: 20px; animation: bounce-and-pulse 60s infinite;'>Your CUD AI Assistant</h4>", unsafe_allow_html=True),
}

model = GoogleGenerativeAI(temperature=0.0,
            model="gemini-pro",
            google_api_key=os.environ["api_key"],
            
        )

#Extracting and Splitting PDF
def extract_text(list_of_uploaded_files):
    pdf_text=''
    for uploaded_pdfs in list_of_uploaded_files:
        read_pdf=PdfReader(uploaded_pdfs)
        for page in read_pdf.pages:
            pdf_text+=page.extract_text()
    
    
    return pdf_text
    
def adjust_final_number(string: str, max_threshold: int, initial_number: int, vectorstore) -> int:
    final_number = initial_number
    while final_number < max_threshold:
        retriever = vectorstore.as_retriever(search_kwargs={"k": final_number})
        docs = retriever.get_relevant_documents(string)
        text = "".join([doc.page_content for doc in docs])
        if num_tokens_from_string(text) < max_threshold:
            final_number += 1
        else:
            break
    return final_number



def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=4000)
    chunks = text_splitter.split_text(text)
    return chunks


#Embedding and storing the pdf Local
def get_embeddings_and_store_pdf(chunk_text):
    if not isinstance(chunk_text, list):
        raise ValueError("Text must be a list of text documents")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        create_embedding = FAISS.from_texts(chunk_text, embedding=embedding_model)
        create_embedding.save_local("embeddings_index")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")

#Generating user response for the pdf
def get_generated_user_input(user_question):
    # Initialize Google Generative AI Embeddings with the specified model
    text_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load stored embeddings using FAISS, allowing dangerous deserialization
    stored_embeddings = FAISS.load_local("embeddings_index", text_embedding, allow_dangerous_deserialization=True)
    
    # Search for similarity between user question and stored embeddings
    check_pdf_similarity = stored_embeddings.similarity_search(user_question)

    # Generate a prompt for answering the query based on the context and user question
    prompt = f"Answer this query based on the Context: \n{check_pdf_similarity}?\nQuestion: \n{user_question}"

    # Send the prompt as a message in the chat history and retrieve the response
    pdf_response = st.session_state.chat_history.send_message(prompt)
    # Return the response
    return pdf_response

#Clearing Chat 
def clear_chat_convo():
    st.session_state.chat_history.history=[]

#Changing Role Names/Icons
def role_name(role):    
    if role == "model":  
        return "bot.png"  
    elif role=='user':
        return 'user.png'
    else:
        return None 

#Text Splits
def stream(response):
    for word in response.text.split(" "):
        yield word + " "
        time.sleep(0.04)

#Extracts the user question from pdf prompt in get_generated_user_input() 
def extract_user_question(prompt_response):
    # Iterate through the parts of the prompt response in reverse order
    for part in reversed(prompt_response):
        # Check if the part contains the keyword "Question:"
        if "Question:" in part.text:
            # Split the text after "Question:" and return the extracted user question
            return part.text.split("Question:")[1].strip()

def main():
    db=Chroma(persist_directory="vectorstore",embedding_function=embeddings)
    # Opening CSS File
    # Read the contents of 'dark.css' file and embed it in the HTML style tag
    with open('dark.css') as f:
        # Apply the CSS style to the page
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 

    # Start a conversation using the model, initially with an empty history
   

    # Check if 'chat_history' is not already in the session state
    if "chat_history" not in st.session_state:
        # If not, initialize 'chat_history' with the start of the conversation
        st.session_state.chat_history = []
    
    # Iterate over each message in the chat history
    for message in st.session_state.chat_history:
        # Get the role name of the message and fetch corresponding avatar if available
        avatar = role_name(message.role)
        # Check if avatar exists
        if avatar:
            # Display the message with the role's avatar
            with st.chat_message(message.role, avatar=avatar):
                # Check if the message has 'content' in its parts
                if "content" in message.parts[0].text: 
                    # Extract the user's question from the message parts (if available)
                    user_question = extract_user_question(message.parts)
                    # Check if a user question is extracted
                    if user_question:
                        # Display the user question using Markdown
                        st.markdown(user_question)
                else:  
                    # If 'content' is not found in the parts, display the message text using Markdown
                    st.markdown(message.parts[0].text)
            
    # Get user input from the chat interface
    user_question = st.chat_input("Ask ChatCUD...")

    # Processing user input
    if user_question is not None and user_question.strip() != "":
        # Display the user input message with user avatar
        with st.chat_message("user", avatar="user.png"):
            st.write(user_question)
            final_number = adjust_final_number(user_question, 15000, 4,db)
            retriever=db.as_retriever(search_kwargs={"k": final_number})
            

            template = """
            Answer the question based only on the following context:
            {context}

            Answer the following question:
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)


            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )
            # Question
            response=rag_chain.invoke(user_question)

            # Pre-load PDFs and extract text from them
            
        
        # If responses are generated
        if response:
            # Display the responses with assistant's avatar
            with st.chat_message("assistant", avatar="bot.png"):
                # Write the responses to the chat
                st.markdown(response)

    # Add a button in the sidebar to clear the chat history
    st.sidebar.button("Click to Clear Chat History", on_click=clear_chat_convo)


if __name__ == "__main__":
    main()