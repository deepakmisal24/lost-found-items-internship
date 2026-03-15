import pandas as pd
import ollama

# The Zero-Fluff, High-Density System Prompt
SYSTEM_PROMPT = """You are an AI data extractor. Your job is to turn formal product descriptions into highly concise, visually-focused search queries or direct item logs.

CRITICAL INSTRUCTIONS:
1. ZERO FLUFF: Do NOT include any conversational filler, emotions, or storytelling (e.g., "I lost my...", "Please help", "I left it...", "I am stressed").
2. KEYWORDS ONLY: Focus entirely on physical attributes: Brand, Model, Color, Material, Shape, and defining features.
3. BE DIRECT: Write in short, punchy phrases or fragments. Add realistic, hyper-specific visual details if the formal description is too vague (like stickers, case colors, scratches).
4. Focus more on BRAND NAME, PRODUCT NAME, COLOR, and MATERIAL. 

following are EXAMPLES OF DESIRED OUTPUT(get only inspiration don't copy paste the same examples):
- "white iphone 9 pro max which has a pink colored transparent backcover with some cute dog stickers on the backcover",
- "Sony brand wireless headphone. black in color with leather ear cups.",
- "sporty sun shades with rectangular frame",etc

Output ONLY the direct description. Nothing else."""

def generate_student_report(name, description):
    """Uses local LLM to generate a zero-fluff, highly descriptive item log."""
    user_prompt = f"Product Name: {name}\nFormal Description: {description}"
    
    try:
        response = ollama.chat(
            model='deepseek-r1:1.5b', 
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"  [!] Local LLM Error for {name[:20]}: {e}")
        return f"{name}. specific physical details unknown."

if __name__ == "__main__":
    print("Loading scraped products...")
    try:
        df = pd.read_csv("scraped_products.csv")
    except FileNotFoundError:
        print("Error: scraped_products.csv not found. Please run the scraper first!")
        exit()
        
    df['lost_description'] = ""
    
    print(f"Generating zero-fluff, high-density item logs for {len(df)} items using gemma3:1b...\n")
    
    for index, row in df.iterrows():
        name = str(row.get('name', 'Unknown Item'))
        desc = str(row.get('description', 'No description available'))
        
        print(f"Processing ({index+1}/{len(df)}): {name[:40]}...")
        
        student_report = generate_student_report(name, desc)
        df.at[index, 'lost_description'] = student_report
        
    output_filename = "simulated_lost_items.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n🎉 Success! Your highly optimized dataset is saved as {output_filename}.")