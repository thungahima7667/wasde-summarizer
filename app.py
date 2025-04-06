import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = OpenAI()

st.title("📊 WASDE Commodity Summarizer")

uploaded_file = st.file_uploader("Upload a WASDE PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])

    # Commodity-aware segmentation
    def extract_commodity_sections(text):
        import re
        commodities = ["corn", "soybeans", "wheat"]
        pattern = r"(?i)(" + "|".join(commodities) + r")\s+[\s\S]+?(?=\n[A-Z])"
        matches = re.findall(pattern, text)
        return {commodity.lower(): section for commodity, section in matches}

    sections = extract_commodity_sections(full_text)

    if sections:
        selected_commodity = st.selectbox("Choose a commodity", list(sections.keys()))

        if selected_commodity:
            section = sections[selected_commodity]

            # Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(section)

            # Embedding
            vectors = embedder.encode(chunks)
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(vectors))

            # Summarize
            context = "\n".join(chunks)
            prompt = f"Summarize key changes for {selected_commodity.upper()} in the following WASDE report section:\n\n{context}"
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("📌 Summary")
            st.write(response.choices[0].message.content)
    else:
        st.warning("No commodity sections found. Try another PDF.")