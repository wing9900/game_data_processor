import openai
import os
import json
import re
from typing import List, Dict, Any, TypedDict

# --- Define the specific type for a Pinecone vector dictionary ---
class PineconeVector(TypedDict):
    id: str
    values: List[float]
    metadata: Dict[str, Any]
    text_content: str

# --- 1. OpenAI Embedding Setup ---
client = openai.OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small", dimensions: int = 1536) -> List[float]:
    """
    Generates an embedding for the given text using a specified OpenAI embedding model.
    Includes basic error handling and prepares for batching later.

    Args:
        text: The input string to embed.
        model: The name of the embedding model to use.
        dimensions: The desired output dimensionality of the embedding.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        Exception: If there's an error during embedding generation.
    """
    if not text or not text.strip():
        print("Warning: Attempted to embed empty or whitespace-only text.")
        return []

    try:
        response = client.embeddings.create(
            input=[text],
            model=model,
            dimensions=dimensions
        )
        return response.data[0].embedding
    except openai.APICallError as e:
        print(f"OpenAI API Error for text '{text[:50]}...': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during embedding for text '{text[:50]}...': {e}")
        raise

# --- 2. Helper to Save Vectors to JSON Lines File (for Pinecone upload) ---
def save_vectors_to_jsonl(vectors: List[PineconeVector], filepath: str = "pinecone_vectors.jsonl"):
    """
    Appends a list of vector dictionaries to a JSON Lines file.
    This file is optimized for Pinecone upload.
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        for vector_dict in vectors:
            f.write(json.dumps(vector_dict) + '\n')
    print(f"Appended {len(vectors)} vectors to {filepath}")

# --- 3. Helper to Write Pretty JSON Output to a File ---
def write_pretty_json_output(
    input_jsonl_filepath: str = "pinecone_vectors.jsonl",
    output_json_filepath: str = "pinecone_vectors_pretty.json",
    remove_embeddings_for_display: bool = True
):
    """
    Reads a .jsonl file, converts its contents into a pretty-printed JSON array,
    and writes it to a new file. Embeddings can be entirely removed for easier review.

    Args:
        input_jsonl_filepath: The path to the .jsonl file generated for Pinecone.
        output_json_filepath: The path where the pretty-printed JSON file will be saved.
        remove_embeddings_for_display: If True, 'values' array will be replaced by a placeholder string.
    """
    if not os.path.exists(input_jsonl_filepath):
        print(f"Error: Input .jsonl file not found at '{input_jsonl_filepath}'")
        return

    all_vectors_data_for_display = []
    with open(input_jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                vector_data: PineconeVector = json.loads(line)

                # --- Modify the vector for display purposes ---
                if remove_embeddings_for_display:
                    vector_data["values"] = "[EMBEDDING_VECTOR_REMOVED_FOR_READABILITY]"

                all_vectors_data_for_display.append(vector_data)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line from {input_jsonl_filepath}: {e}\nContent: {line.strip()}")
                break

    with open(output_json_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_vectors_data_for_display, f, indent=2, ensure_ascii=False)

    print(f"\nPretty-printed data saved to '{output_json_filepath}'.")

# --- NEW HELPER: Aggressive Character Scrubbing for Raw Content ---
def _scrub_webpage_content_chars(text: str) -> str:
    """
    Performs aggressive character scrubbing to remove problematic non-standard
    or invisible characters that might cause parsing issues.
    """
    # 1. Convert to ASCII and ignore non-ASCII characters
    cleaned_text = text.encode('ascii', 'ignore').decode('ascii')

    # 2. Normalize hyphens and common dashes to standard hyphen-minus
    cleaned_text = cleaned_text.replace('–', '-').replace('—', '-')

    # 3. Remove zero-width spaces and similar invisible characters
    cleaned_text = cleaned_text.replace('\u200B', '').replace('\u200C', '').replace('\u200D', '').replace('\uFEFF', '')

    # 4. Remove any remaining non-standard whitespace or control characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)

    return cleaned_text

# --- 5. Function to Read and Clean Webpage Content from File ---
# This function is not called in the main block below, as we are using the hardcoded string directly.
# However, it's kept here if you later decide to read content from an actual file.
def read_and_clean_webpage(filepath: str) -> str:
    """
    Reads content from a text file and performs initial cleaning steps
    to remove common webpage junk elements.

    Args:
        filepath: The path to the .txt file containing the webpage content.

    Returns:
        The cleaned string content of the webpage.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Webpage file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- Apply aggressive character scrubbing first ---
    content = _scrub_webpage_content_chars(content)

    # --- Existing Cleaning Logic ---
    # 1. Remove "SIGN IN TO EDIT" line and potentially related elements (e.g., chat bubble info)
    content = re.sub(r'\s*\d+\s*[\u200B-\u200D\uFEFF]?\s*\uD83D[\uDCAD\uDCE4\uDCAC\uDD8E\uDD81-\uDD8E\u200B-\u200D\uFEFF]?[^\n]*?SIGN IN TO EDIT.*?[\u200B-\u200D\uFEFF]?\u22EE\s*', '', content, flags=re.DOTALL)
    content = re.sub(r'SIGN IN TO EDIT.*', '', content)

    # 2. Remove "Contents [hide]" block (Table of Contents)
    content = re.sub(r'Contents\s*\[hide\].*?(?=(Overview|\d+\s*Overview|\d+\s*Stats|\d+\s*Firepower|\d+\s*Speed|\d+\s*Health))', '', content, flags=re.DOTALL | re.IGNORECASE)

    # 3. Remove standalone numbers/decimals on lines (like 1, 2, 2.1 from TOC)
    content = re.sub(r'^\s*\d+(\.\d+)?\s*$', '', content, flags=re.MULTILINE)

    # 4. Remove excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()

# --- Parsing Function for Spitfire ---
def parse_spitfire_webpage_content(webpage_text: str) -> List[PineconeVector]:
    """
    Parses the cleaned text content of a Spitfire webpage
    into structured chunks for Pinecone.

    Args:
        webpage_text: The pre-cleaned content of the Spitfire webpage.

    Returns:
        A list of dictionaries, each representing a Pinecone vector.
    """
    processed_vectors: List[PineconeVector] = []
    item_name = "Spitfire"
    entity_type = "aircraft"

    # --- Chunk 1: General Information ---
    general_info_text = (
        f"The Supermarine Spitfire is a World War 2-era British fighter plane, retired in 1954. "
        f"It is automatically unlocked after reaching Player Level 20 from Daily Challenges and "
        f"found in the Plane Hangar after Rebirth 7. "
        f"It has 1 seating capacity, 1 hull, 1 weapon, 1 engine, and the utility 'Zoom In'."
    )
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_general_info",
        values=get_embedding(general_info_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "general_info",
            "unlock_method": "Player Level 20",
            "unlock_level": 20, # Specific level for filtering
            "hangar_access_rebirth_level": 7, # For context if they ask about hangar
            "seating_capacity": 1,
            "hulls": 1,
            "weapons_count": 1,
            "engines_count": 1,
            "utility": ["Zoom In"]
        },
        text_content=general_info_text
    ))

    # --- Chunk 2: Full Overview Text ---
    full_overview_text = """The Spitfire is a heavily underrated plane in War Tycoon, being under the assumption of being extremely weak due to its status as a propeller plane. The plane is famous for its rapid turn rate even for its age, being able to out-turn modern aircraft like the F-14 Tomcat and the A-10 Warthog, making these jets extremely vulnerable to the high-damage cannons on the Spitfire.
The armament diminishes shields, aircraft, and even tanks, taking as little as two dives to completely destroy a full HP MAUS, being the tank with the highest health in the game, and being very difficult for planes and helicopters to destroy if not already low on HP. What makes it even more overpowered is the very long fire rate it has; it takes more than 10 seconds to go into cooldown and recharges rapidly if you stop firing the cannons before it hits the cooldown. This allows for a practically infinite source of ammo, and while your opponents are waiting for their cooldown to finish, they become an easy target to eliminate.
The only true weakness of the Spitfire is its slow speed compared to the other aircraft and its role as a propeller plane, not being able to chase down enemies across the map without veteran aim or vertically up into the sky, where it will stall and lose altitude. It is important that when in a dogfight with the Spitfire, you have as low an altitude as possible and have room to go full speed so you can climb up, fly fast and dive back on the Spitfire when far enough away from it.
Overall, this plane may take some practice to use correctly, for it can easily be defeated by modern aircraft, for it has unsurprisingly low HP compared to them, but it's worth noting that this plane is a large threat you'll need to confront in the skies.
Stats
Firepower
Armament
Damage Per Shot (Non-Upgraded)
Damage Per Shot (Tier 1)
Damage Per Shot (Tier 2)
Damage Per Shot (Tier 3)
20mm Cannons
40 Damage
44 Damage
48 Damage
52 Damage
.303 Browning Machine Gun
20 Damage
22 Damage
24 Damage
26 Damage
Speed
Speed (Non-Upgraded)
Speed (Tier 1)
Speed (Tier 2)
Speed (Tier 3)
205 MPH
225 MPH
246 MPH
266 MPH
Health
Health (Non-Upgraded)
Health (Tier 1)
Health (Tier 2)
Health (Tier 3)
750 HP
825 HP
900 HP
975 HP
"""

    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_overview_full_text",
        values=get_embedding(full_overview_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "overview_full_text",
            "section_title": "Overview"
        },
        text_content=full_overview_text
    ))

    # --- Chunk 3: Concise Overview/Summary ---
    concise_overview_text = (
        f"The {item_name} is an underrated propeller plane known for its rapid turn rate, "
        f"out-turning modern jets like F-14 and A-10. Its high-damage cannons can destroy even a full HP MAUS in two dives. "
        f"It has a very long fire rate before cooldown, allowing practically infinite ammo. "
        f"Its main weakness is slow speed compared to other aircraft, making long chases or vertical climbs difficult, "
        f"and it has low HP compared to modern aircraft."
    )
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_overview_concise",
        values=get_embedding(concise_overview_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "overview_summary",
            "strengths": ["Rapid turn rate", "Out-turns modern aircraft (F-14, A-10)", "High-damage cannons", "Effective against shields, aircraft, tanks (e.g., MAUS)", "Long fire rate before cooldown", "Practically infinite ammo"],
            "weaknesses": ["Slow speed", "Propeller plane limitations (chasing, vertical climb)", "Can stall/lose altitude high up", "Low HP compared to modern aircraft"],
            "role": "Underrated fighter",
            "playstyle_notes": "Best at low altitude, full speed for climbs/dives in dogfights."
        },
        text_content=concise_overview_text
    ))

    # --- Chunk 4: Armament - 20mm Cannons ---
    cannons_armament_text = "Equipped with two 20mm cannons. These high-damage cannons are effective against shields, aircraft, and tanks (can destroy a full HP MAUS in two dives). They have a very long fire rate before cooldown (more than 10 seconds) and recharge rapidly."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_20mm_cannons",
        values=get_embedding(cannons_armament_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": "20mm Cannons",
            "count": 2,
            "damage_notes": "high-damage, effective against shields/aircraft/tanks (e.g., MAUS)",
            "fire_rate_notes": "very long fire rate (over 10s before cooldown), rapid recharge"
        },
        text_content=cannons_armament_text
    ))

    # --- Chunk 5: Armament - .303 Browning Machine Guns ---
    mg_armament_text = "Comes with four .303 Browning Machine Guns. Provides additional damage."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_303_browning_mg",
        values=get_embedding(mg_armament_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": ".303 Browning Machine Gun",
            "count": 4,
            "damage_notes": "additional damage"
        },
        text_content=mg_armament_text
    ))

    # --- Chunk 6: Stats - Speed ---
    speed_text = "Speed (Non-Upgraded): 205 MPH, Speed (Tier 1): 225 MPH, Speed (Tier 2): 246 MPH, Speed (Tier 3): 266 MPH."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_stats_speed",
        values=get_embedding(speed_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "stats",
            "stat_type": "Speed",
            "unit": "MPH",
            "tiers": {
                "Non-Upgraded": 205,
                "Tier 1": 225,
                "Tier 2": 246,
                "Tier 3": 266
            }
        },
        text_content=speed_text
    ))

    # --- Chunk 7: Stats - Health ---
    health_text = "Health (Non-Upgraded): 750 HP, Health (Tier 1): 825 HP, Health (Tier 2): 900 HP, Health (Tier 3): 975 HP."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_stats_health",
        values=get_embedding(health_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "stats",
            "stat_type": "Health",
            "unit": "HP",
            "tiers": {
                "Non-Upgraded": 750,
                "Tier 1": 825,
                "Tier 2": 900,
                "Tier 3": 975
            }
        },
        text_content=health_text
    ))

    # --- Chunk 8: Stats - Firepower (20mm Cannons) ---
    firepower_20mm_text = "20mm Cannons Damage Per Shot: Non-Upgraded: 40 Damage, Tier 1: 44 Damage, Tier 2: 48 Damage, Tier 3: 52 Damage."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_stats_firepower_20mm_cannons",
        values=get_embedding(firepower_20mm_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "stats",
            "stat_type": "Firepower",
            "armament": "20mm Cannons",
            "unit": "Damage",
            "tiers": {
                "Non-Upgraded": 40,
                "Tier 1": 44,
                "Tier 2": 48,
                "Tier 3": 52
            }
        },
        text_content=firepower_20mm_text
    ))

    # --- Chunk 9: Stats - Firepower (.303 Browning Machine Gun) ---
    firepower_303_text = ".303 Browning Machine Gun Damage Per Shot: Non-Upgraded: 20 Damage, Tier 1: 22 Damage, Tier 2: 24 Damage, Tier 3: 26 Damage."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_stats_firepower_303_mg",
        values=get_embedding(firepower_303_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "stats",
            "stat_type": "Firepower",
            "armament": ".303 Browning Machine Gun",
            "unit": "Damage",
            "tiers": {
                "Non-Upgraded": 20,
                "Tier 1": 22,
                "Tier 2": 24,
                "Tier 3": 26
            }
        },
        text_content=firepower_303_text
    ))

    # --- Chunk 10: Category Information for this Plane ---
    category_text = f"The {item_name} is classified under the 'Fighters' category of planes."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_category_membership",
        values=get_embedding(category_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "category_membership",
            "aircraft_category": "Fighters"
        },
        text_content=category_text
    ))

    return processed_vectors


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure the output file is empty or doesn't exist to start fresh
    # This will clear the file before each run, ensuring only the currently processed data is included.
    # We will output to a specific file for Spitfire now
    spitfire_output_jsonl = "pinecone_spitfire_vectors.jsonl"
    spitfire_output_pretty_json = "pinecone_spitfire_vectors_pretty.json"

    if os.path.exists(spitfire_output_jsonl):
        os.remove(spitfire_output_jsonl)
        print(f"Removed existing {spitfire_output_jsonl} to start fresh.")

    # --- Spitfire Webpage Content ---
    spitfire_webpage_content = r"""
The Supermarine Spitfire is a World War 2-era British fighter plane, retired from service in the Royal Air Force in 1954.
It is automatically unlocked after reaching Player Level 20 from the Daily Challenges and can be found in the Plane Hangar after reaching Rebirth 7.
Overview
Spitfire

The in-game render of the Spitfire.
General Information
Price
Player Level 20
(Requires Player Level 20 to be reached.)
Speed (Minimum)
Speed (Maximum)
205 MPH
266 MPH

Health (Minimum)
Health (Maximum)
750 HP
975 HP
Armament
- 2x 20mm Cannons;
- 4x .303 Browning Machine Guns
Utility
- Zoom In
Seating Capacity
1
Hulls
Weapons
Engines
1
1
1
The Spitfire is a heavily underrated plane in War Tycoon, being under the assumption of being extremely weak due to its status as a propeller plane. The plane is famous for its rapid turn rate even for its age, being able to out-turn modern aircraft like the F-14 Tomcat and the A-10 Warthog, making these jets extremely vulnerable to the high-damage cannons on the Spitfire.
The armament diminishes shields, aircraft, and even tanks, taking as little as two dives to completely destroy a full HP MAUS, being the tank with the highest health in the game, and being very difficult for planes and helicopters to destroy if not already low on HP. What makes it even more overpowered is the very long fire rate it has; it takes more than 10 seconds to go into cooldown and recharges rapidly if you stop firing the cannons before it hits the cooldown. This allows for a practically infinite source of ammo, and while your opponents are waiting for their cooldown to finish, they become an easy target to eliminate.
The only true weakness of the Spitfire is its slow speed compared to the other aircraft and its role as a propeller plane, not being able to chase down enemies across the map without veteran aim or vertically up into the sky, where it will stall and lose altitude. It is important that when in a dogfight with the Spitfire, you have as low an altitude as possible and have room to go full speed so you can climb up, fly fast and dive back on the Spitfire when far enough away from it.
Overall, this plane may take some practice to use correctly, for it can easily be defeated by modern aircraft, for it has unsurprisingly low HP compared to them, but it's worth noting that this plane is a large threat you'll need to confront in the skies.
Stats
Firepower
Armament
Damage Per Shot (Non-Upgraded)
Damage Per Shot (Tier 1)
Damage Per Shot (Tier 2)
Damage Per Shot (Tier 3)
20mm Cannons
40 Damage
44 Damage
48 Damage
52 Damage
.303 Browning Machine Gun
20 Damage
22 Damage
24 Damage
26 Damage
Speed
Speed (Non-Upgraded)
Speed (Tier 1)
Speed (Tier 2)
Speed (Tier 3)
205 MPH
225 MPH
246 MPH
266 MPH
Health
Health (Non-Upgraded)
Health (Tier 1)
Health (Tier 2)
Health (Tier 3)
750 HP
825 HP
900 HP
975 HP
"""
    print("\n--- Processing Spitfire ---")

    cleaned_spitfire_content = _scrub_webpage_content_chars(spitfire_webpage_content)
    cleaned_spitfire_content = re.sub(r'\s*\d+\s*[\u200B-\u200D\uFEFF]?\s*\uD83D[\uDCAD\uDCE4\uDCAC\uDD8E\uDD81-\uDD8E\u200B-\u200D\uFEFF]?[^\n]*?SIGN IN TO EDIT.*?[\u200B-\u200D\uFEFF]?\u22EE\s*', '', cleaned_spitfire_content, flags=re.DOTALL)
    cleaned_spitfire_content = re.sub(r'SIGN IN TO EDIT.*', '', cleaned_spitfire_content)
    cleaned_spitfire_content = re.sub(r'Contents\s*\[hide\].*?(?=(Overview|\d+\s*Overview|\d+\s*Stats|\d+\s*Firepower|\d+\s*Speed|\d+\s*Health))', '', cleaned_spitfire_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned_spitfire_content = re.sub(r'^\s*\d+(\.\d+)?\s*$', '', cleaned_spitfire_content, flags=re.MULTILINE)
    cleaned_spitfire_content = re.sub(r'\n{3,}', '\n\n', cleaned_spitfire_content)
    cleaned_spitfire_content = cleaned_spitfire_content.strip()


    spitfire_vectors = parse_spitfire_webpage_content(cleaned_spitfire_content)
    print(f"Generated {len(spitfire_vectors)} vectors for Spitfire.")
    save_vectors_to_jsonl(spitfire_vectors, filepath=spitfire_output_jsonl) # Save to a specific file
    write_pretty_json_output(input_jsonl_filepath=spitfire_output_jsonl, output_json_filepath=spitfire_output_pretty_json)

    print(f"\nProcessing complete. Check '{spitfire_output_jsonl}' (for Pinecone) and '{spitfire_output_pretty_json}' (for review) files.")