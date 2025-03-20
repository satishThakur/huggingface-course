from sentence_transformers import SentenceTransformer
model_name = "all-MiniLM-L6-v2" # Load pre-trained model
embedder = SentenceTransformer(model_name)
sentence = "what are embeddings?"
embeddings = embedder.encode([sentence])
print(embeddings)
print(embeddings.shape)
print(embedder.get_sentence_embedding_dimension())
