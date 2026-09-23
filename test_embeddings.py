from src.embeddings import embedding_store
from src.document_loader import load_knowledge_base

chunks = load_knowledge_base()
print(f'Loaded {len(chunks)} chunks')
embedding_store.build_index(chunks)
print(f'Built index: {len(embedding_store.chunks)} chunks, embeddings shape: {embedding_store.embeddings.shape}')

# Test search
result = embedding_store.search('return policy', top_k=3)
print(f'Search results: {len(result.chunks)} chunks')
for i, chunk in enumerate(result.chunks):
    print(f'  {i+1}. {chunk.metadata.source_file} > {chunk.heading} (score: {result.scores[i]:.3f})')