import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

def build_vector_database():
    input_file = "lost_found_dataset_cleaned.csv"
    db_path = "./chroma_db"
    
    print(f"1. Loading cleaned dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run Phase 2 first!")
        return

    print("2. Loading CLIP Multi-Modal Model...")
    model = SentenceTransformer('clip-ViT-B-32')

    print("3. Initializing ChromaDB with Cosine Similarity...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Create TWO collections to handle text and image searches perfectly
    for name in ["lost_items_text", "lost_items_image"]:
        try:
            chroma_client.delete_collection(name=name)
        except Exception:
            pass
            
    # Using Cosine similarity is much more accurate for CLIP embeddings than the default L2
    text_collection = chroma_client.create_collection(name="lost_items_text", metadata={"hnsw:space": "cosine"})
    image_collection = chroma_client.create_collection(name="lost_items_image", metadata={"hnsw:space": "cosine"})

    print(f"4. Embedding and storing {len(df)} items...\n")
    
    for index, row in df.iterrows():
        first_image_path = str(row['image_paths']).split(',')[0].strip()
        item_id = str(row['product_id'])
        search_text = str(row['searchable_text'])
        
        try:
            # Prepare metadata
            metadata = {
                "name": str(row['name']),
                "category": str(row['category']),
                "price": str(row['price']),
                "image_paths": str(row['image_paths'])
            }
            
            # 1. Embed and store the TEXT
            text_embedding = model.encode(search_text).tolist()
            text_collection.add(
                ids=[item_id], embeddings=[text_embedding], documents=[search_text], metadatas=[metadata]
            )
            
            # 2. Embed and store the IMAGE
            if os.path.exists(first_image_path):
                img = Image.open(first_image_path)
                image_embedding = model.encode(img).tolist()
                image_collection.add(
                    ids=[item_id], embeddings=[image_embedding], documents=[search_text], metadatas=[metadata]
                )
                
            print(f"  ✅ Added to DB: {row['name'][:30]}...")
            
        except Exception as e:
            print(f"  [!] Failed to process item {item_id}: {e}")

    print("\n--- PHASE 3 COMPLETE ---")
    print(f"Vector Database successfully built and saved to: {db_path}")

if __name__ == "__main__":
    build_vector_database()