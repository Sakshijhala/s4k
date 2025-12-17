import numpy as np
import pandas as pd
from txtai.embeddings import Embeddings

np.random.seed(1)

# Load data
df = pd.read_csv("seth-data.csv").dropna()

# IMPORTANT: same columns & same order as Streamlit
titles = df.title.values

# Create embeddings model
embeddings = Embeddings({
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

# Index using dataframe index positions
embeddings.index([
    (i, title, None) for i, title in enumerate(titles)
])

# Save embeddings
embeddings.save("embeddings_seth.tar.gz")

print("✅ embeddings_seth.tar.gz created successfully")

