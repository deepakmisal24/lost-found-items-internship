import os
import requests
import pandas as pd

IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(img_url, img_name):
    """Downloads an image and saves it to the images folder."""
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(IMAGE_DIR, img_name)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return file_path
    except Exception as e:
        pass 
    return None

def fetch_dummy_products():
    """Fetches products from DummyJSON, strictly requiring 4 images."""
    print("Fetching product data from DummyJSON...")
    
    # Requesting 200 items to ensure we get enough valid ones after filtering 
    url = "https://dummyjson.com/products?limit=200" 
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to fetch data from API.")
        return []

    products_data = []
    raw_products = response.json().get('products', [])
    
    for idx, item in enumerate(raw_products):
        name = item.get('title', 'Unknown')
        description = item.get('description', '')
        price = item.get('price', 0)
        
        # Grab the full list of images
        available_images = item.get('images', [])
        
        # --- THE GATEKEEPER ---
        # Only proceed if the product has AT LEAST 3 images 
        if len(available_images) >= 3:
            
            # Slice to get exactly 4 images
            target_images = available_images[:4]
            img_paths = []
            
            for img_idx, img_url in enumerate(target_images):
                img_name = f"product_{idx}_img_{img_idx}.jpg"
                saved_path = download_image(img_url, img_name)
                
                if saved_path:
                    img_paths.append(saved_path)
            
            # Final safety check: Did all 3 actually download?
            if len(img_paths) >= 3:
                products_data.append({
                    "product_id": f"PRD_{idx}",
                    "name": name,
                    "description": description,
                    "price": price,
                    "image_paths": ", ".join(img_paths)
                })
                print(f"✅ Saved: {name} (Got all 3 images)")
            else:
                print(f"⚠️ Skipped: {name} (A download failed)")
                
        else:
            print(f"⏭️ Skipped: {name} (Only has {len(available_images)} images)")

    return products_data

if __name__ == "__main__":
    data = fetch_dummy_products()
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv("scraped_products.csv", index=False)
        print(f"\n🎉 Success! Saved {len(df)} strictly formatted items to scraped_products.csv.") 
    else:
        print("No data was processed.")