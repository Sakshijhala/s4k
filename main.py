import txtai
import numpy as np
import pandas as pd

np.random.seed(1)

df = pd.read_csv('seth-data.csv').dropna()
content = df.content_plain.values

embeddings = txtai.embeddings({
    'path': 'sentence-transformers/all-MiniLM-L6-v2'
    
})
#embeddings.load('embedding.tar.gz')
embeddings.index(content)
embeddings.save('embedding_seth.tar.gz')

