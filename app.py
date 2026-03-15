import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from PIL import Image
import os

st.set_page_config(page_title="Lost & Found Reunion", layout="wide")

# These MUST match the categories we defined in prep_data.py exactly
VALID_CATEGORIES = {
    "Apple Devices (iPhones & MacBooks)": ["iphone", "macbook", "apple airpods","apple"],
    "Android Smartphones": ["samsung universe", "oppo", "huawei", "infinix","smartphone", "galaxy"],
    "Windows Laptops & Surface Devices": ["surface pro", "hp pavilion", "galaxy book", "inbook","laptop","book"],
    "Fragrances & Perfume Oils": ["perfume oil", "brown perfume", "scent xpressio", "eau de perfume"],
    "Facial Skincare & Serums": ["hyaluronic acid", "tree oil", "moisturizer", "freckle treatment", "serum"],
    "Pantry Groceries & Baking": ["daal masoor", "macaroni", "baking powder", "cereal", "orange essence"],
    "Home Decor & Wooden Crafts": ["plant hanger", "wooden bird", "home decoration", "handcraft"],
    "Women's Handbags & Purses": ["handbag for girls", "women purse", "clutch", "bag"],
    "Women's Dresses & Apparel": ["dress", "skirt", "women clothing", "ladies wear"],
    "Men's Shirts & Apparel": ["t-shirt", "half sleeve", "men clothing", "hoodie"],
    "Men's Footwear & Sneakers": ["sneaker", "men shoe", "sports shoe"],
    "Women's Footwear": ["women shoe", "heels", "sandals", "stiletto"],
    "Men's & Women's Watches": ["fossil watch", "rolex", "luxury watch", "chronograph"],
    "Sunglasses & Eyewear": ["sunglasses", "aviator", "shades", "glasses"],
    "Automotive & Motorcycle Parts": ["o-ring", "spark plug", "brake", "motorcycle", "car part","car","vehicle"],
    "Home Lighting & Lamps": ["lamp", "bulb", "lighting", "chandelier"],
    "Furniture & Bedding": ["bed", "sofa", "wooden chair", "furniture"]
}

@st.cache_resource
def load_system():
    text_model = SentenceTransformer('clip-ViT-B-32')
    image_model = SentenceTransformer('clip-ViT-B-32')
    client = chromadb.PersistentClient(path="./chroma_db")
    text_coll = client.get_collection(name="lost_items_text")
    image_coll = client.get_collection(name="lost_items_image")
    return text_model, image_model, text_coll, image_coll

try:
    text_model, image_model, text_collection, image_collection = load_system()
except Exception as e:
    st.error("Could not load the database. Please run build_vector_db.py first!")
    st.stop()

def predict_category(user_query):
    """Uses the LLM to route the query to a specific category BEFORE searching."""
    prompt = f"""You are a strict data router. analyze the user query and find the general category it would belong to example: if user input is 'shirt' then would belong to "men's clothing category"
        Analyze this user query: "{user_query}"
        Choose the single most accurate category from this exact list: {VALID_CATEGORIES}

        CRITICAL: Output ONLY the exact category string from the list. Do not add quotes, punctuation, or conversational text."""
    
    try:
        response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': prompt}])
        predicted = response['message']['content'].strip()
        
        # Safety check: Make sure the LLM didn't hallucinate a fake category
        for valid_cat in VALID_CATEGORIES:
            if valid_cat.lower() in predicted.lower():
                return valid_cat
        return None # Fallback if the LLM gets confused
    except Exception:
        return None

def explain_match(user_query, item_name, item_desc):
    prompt = f"""A student lost: "{user_query}"
We found: "{item_name}" - Details: "{item_desc}"
In 2 brief sentences, explain why this inventory item is a match."""
    try:
        response = ollama.chat(model='gemma3:1b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception:
        return "I think this might be your item based on the database match!"

st.title("🔍 Lost & Found Reunion")
st.write("A Multi-Modal Semantic Search Engine with Pre-Filtering")

tab1, tab2 = st.tabs(["Search by Text", "Search by Image"])

def display_results(results, user_query):
    # Check if results actually contain data
    if not results['ids'] or not results['ids'][0]:
        st.warning("No matches found in this category.")
        return

    st.subheader("Top Matches")
    cols = st.columns(3)
    
    for idx, col in enumerate(cols):
        if idx < len(results['ids'][0]):
            metadata = results['metadatas'][0][idx]
            distance = results['distances'][0][idx]
            
            confidence = max(0.0, min(100.0, (1.0 - distance) * 100))
            
            with col:
                st.markdown(f"### Match #{idx+1}")
                st.progress(int(confidence)/100, text=f"Confidence: {confidence:.1f}%")
                
                img_path = metadata['image_paths'].split(',')[0]
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                
                st.write(f"**Item:** {metadata['name']}")
                st.write(f"**Category:** {metadata['category']}")
                
                with st.spinner("Generating explanation..."):
                    explanation = explain_match(user_query, metadata['name'], results['documents'][0][idx])
                st.info(explanation)

# --- TAB 1: TEXT SEARCH (Upgraded with Routing) ---
with tab1:
    text_query = st.text_input("Describe the item you lost (e.g., 'Apple Iphone')")
    if st.button("Search Database", key="text_btn"):
        if text_query:
            # STEP 1: Predict Category
            with st.spinner("Classifying query..."):
                target_category = predict_category(text_query)
            
            with st.spinner(f"Searching vectors inside '{target_category or 'All Categories'}'..."):
                query_embedding = text_model.encode(text_query).tolist()
                
                # STEP 2: Filtered Semantic Search
                if target_category:
                    # Search ONLY where category matches exactly
                    results = text_collection.query(
                        query_embeddings=[query_embedding], 
                        n_results=3,
                        where={"category": target_category} 
                    )
                else:
                    # Fallback to global search if category prediction fails
                    results = text_collection.query(
                        query_embeddings=[query_embedding], 
                        n_results=3
                    )
                
                display_results(results, text_query)
        else:
            st.warning("Please enter a description first.")

# --- TAB 2: IMAGE SEARCH (Kept visual-first) ---
with tab2:
    uploaded_file = st.file_uploader("Upload a photo of your lost item", type=['jpg', 'jpeg', 'png'])
    if st.button("Search via Image", key="img_btn"):
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Your Uploaded Image", width=250)
            
            with st.spinner("Analyzing image vectors..."):
                query_embedding = image_model.encode(img).tolist()
                results = image_collection.query(query_embeddings=[query_embedding], n_results=3)
                display_results(results, "an uploaded photo")