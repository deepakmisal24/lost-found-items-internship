import pandas as pd
import os

def assign_category(name, description=""):
    """An upgraded categorizer that scans both name and description against a broad keyword map."""
    # Combine name and description so we don't miss context
    text = str(name).lower() + " " + str(description).lower()
    
    # A robust dictionary mapping for real-world items
# A hyper-specific dictionary mapping for the exact DummyJSON inventory
    category_map = {
        "Apple Devices (iPhones & MacBooks)": ["iphone", "macbook", "apple airpods"],
        "Android Smartphones": ["samsung universe", "oppo", "huawei", "infinix smartphone", "galaxy"],
        "Windows Laptops & Surface Devices": ["surface pro", "hp pavilion", "galaxy book", "inbook"],
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
        "Automotive & Motorcycle Parts": ["o-ring", "spark plug", "brake", "motorcycle", "car part"],
        "Home Lighting & Lamps": ["lamp", "bulb", "lighting", "chandelier"],
        "Furniture & Bedding": ["bed", "sofa", "wooden chair", "furniture"]
    }
    
    # Check our text against the dictionary
    for category, keywords in category_map.items():
        if any(word in text for word in keywords):
            return category
            
    return 'Miscellaneous - Other'

def prepare_dataset():
    input_file = "simulated_lost_items.csv"
    output_file = "lost_found_dataset_cleaned.csv"
    
    print(f"Loading raw dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Make sure you finished Phase 1!")
        return

    initial_count = len(df)

    # 1. Remove Duplicates & Broken Data
    df.drop_duplicates(subset=['name'], inplace=True)
    df.dropna(subset=['lost_description', 'image_paths'], inplace=True)
    
    # 2. Standardize Categories
    df['category'] = df.apply(lambda row: assign_category(row['name'], row['description']), axis=1)
    
    # 3. Create Combined Searchable Text
    # This combines the panicked student report with the broad category context
    # This single string is what we will feed to the Sentence-Transformer in Phase 3
    df['searchable_text'] = (
        "Category: " + df['category'] + ". " + 
        "Student Report: " + df['lost_description']
    )
    
    # 4. Filter out rows where image files might have gone missing locally
    valid_rows = []
    for index, row in df.iterrows():
        # Check if the first image in the comma-separated list actually exists
        first_image = str(row['image_paths']).split(',')[0].strip()
        if os.path.exists(first_image):
            valid_rows.append(True)
        else:
            print(f"  [!] Dropping item '{row['name'][:20]}...' - Image not found locally: {first_image}")
            valid_rows.append(False)
            
    df = df[valid_rows]

    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    
    final_count = len(df)
    print("\n--- PHASE 2 COMPLETE ---")
    print(f"Started with: {initial_count} items")
    print(f"Cleaned and validated: {final_count} items")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    prepare_dataset()