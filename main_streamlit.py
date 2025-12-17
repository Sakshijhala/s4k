import streamlit as st
import numpy as np
import pandas as pd
from txtai.embeddings import Embeddings

@st.cache
def load_data_and_embeddings():
    np.random.seed(1)

    df = pd.read_csv("seth-data.csv").dropna()

    # Make sure TITLE exists
    titles = df.title.values
    urls = df.url.values

    embeddings = Embeddings({
        "path": "sentence-transformers/all-MiniLM-L6-v2"
    })

    embeddings.load("embeddings_seth.tar.gz")

    return titles,urls, embeddings

titles,urls, embeddings = load_data_and_embeddings()

st.title("Seth Blog Search Engine")

query = st.text_input("Enter a search query:")

if st.button("Search"):
    if query.strip():
        results = embeddings.search(query, 5)

        actual_results = [f'Title:{titles[x[0]]}, URL : {urls[x[0]]}'for x in results]

        for res in actual_results:
            st.write(res)
    else:
        st.warning("Please enter a query")
