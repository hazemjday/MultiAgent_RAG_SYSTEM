from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

sentences = ["Lionel Messi is a footballer.", "Cristiano Ronaldo is his rival."]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"Embeddings shape: {embeddings.shape}")