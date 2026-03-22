from helper import helper
# Test kodu
embeddings = helper.download_hugging_face_embeddings()
vector_db = helper.loadVectors(embeddings)

query = "dropshipping"
docs = vector_db.similarity_search(query, k=3)

for i, doc in enumerate(docs):
    print(f"\nSonuç {i+1}:")
    print(doc.page_content[:200] + "...")