import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.google import GoogleGenerativeAIEmbeddings
from langchain.chat_models.google import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st


# Configure Google Generative AI API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


# Cache data functions to optimize loading
@st.cache_data
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    if pdf_reader.is_encrypted:
        try:
            pdf_reader.decrypt("")
        except Exception as e:
            print(f"Failed to decrypt PDF: {e}")
            return text

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


@st.cache_data
def get_text_chunks_with_metadata(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    # Adding metadata
    return [(chunk, {"source": f"Chunk {i+1}"}) for i, chunk in enumerate(chunks)]


@st.cache_resource
def load_or_create_vector_store(pdf_path):
    if os.path.exists("faiss_index"):
        # Load existing vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # Create new vector store
        raw_text = get_pdf_text(pdf_path)
        text_chunks_with_metadata = get_text_chunks_with_metadata(raw_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_texts([chunk for chunk, _ in text_chunks_with_metadata],
                                        embedding=embeddings,
                                        metadata=[metadata for _, metadata in text_chunks_with_metadata])
        vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.3,
                                   google_api_key=GOOGLE_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to handle user input and generate response
def user_input(user_question, chain, vector_store, chat_history):
    docs = vector_store.similarity_search(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Extract metadata (e.g., source context)
    metadata = "Sources: " + ", ".join([doc.metadata.get("source", "Unknown") for doc in docs])

    # Append the question, answer, and metadata to the chat history
    chat_history.append((user_question, response['output_text'], metadata))

    # Display the entire chat history
    for i, (question, answer, metadata) in enumerate(chat_history):
        st.write("---")
        st.write(f"**:bust_in_silhouette: You ({i + 1}):** {question}", unsafe_allow_html=True)
        st.write(f"**:robot_face: AI ({i + 1}):** {answer}", unsafe_allow_html=True)
        st.write(f"**Source** {metadata}", unsafe_allow_html=True)


# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Pakistan Constitution Assistance", layout="centered",
                       initial_sidebar_state="expanded", page_icon="💚")

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat with Pakistan's Constitution</h1>",
                unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #555;'>Welcome! This tool allows you to interactively ask questions about Pakistan's Constitution.</p>
        """, unsafe_allow_html=True)

    # Initialize chat history list in Streamlit session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load or create the vector store
    pdf_path = "pdfs/PakConstitution.pdf"
    vector_store = load_or_create_vector_store(pdf_path)

    # Initialize the conversational chain
    chain = get_conversational_chain()

    # Sidebar with team information and Pakistan flag
    with st.sidebar:
        st.markdown("<h2 style='color: #4CAF50;'>Team Info</h2>", unsafe_allow_html=True)
        st.markdown("""
        **Team Name:** LegalTech Wizards
        - [Ghulam Mahiyudin](https://www.linkedin.com/in/ghulammahiyudin)
        - [Safa Yousaf](https://www.linkedin.com/in/safa-yousaf-6b2290308)
        - [Muqadas Zahra](https://www.linkedin.com/in/muqadas-zahra-404891283 )
        """, unsafe_allow_html=True)

    # Text input for user question with a nice placeholder
    user_question = st.text_input("Ask a Question:", placeholder="Type your question about the Constitution here...")

    # Button to get response
    if st.button("Get Response"):
        if user_question:
            with st.spinner("Generating response..."):
                user_input(user_question, chain, vector_store, st.session_state.chat_history)
            st.balloons()

    st.write("---")

    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p style='margin: 0;'> We encourage you to verify information from the official Constitution of Pakistan available at <a href="https://na.gov.pk/uploads/documents/1333523681_951.pdf" target="_blank" style='color: #007bff;'>this link</a>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: center; color: #888;'>Design & Developed with 💚</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
