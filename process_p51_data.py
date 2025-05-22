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
# This function is duplicated for convenience in each script, but could be put in a shared utility file.
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

# --- Parsing Function for P-51 Mustang ---
def parse_p51_webpage_content(webpage_text: str) -> List[PineconeVector]:
    """
    Parses the cleaned text content of a P-51 Mustang-like webpage
    into structured chunks for Pinecone.
    """
    processed_vectors: List[PineconeVector] = []
    item_name = "P-51 Mustang"
    entity_type = "aircraft"

    # --- Chunk 1: General Information ---
    general_info_text = (
        f"The North American P-51 Mustang (referred to as simply the 'P-51 Mustang' in War Tycoon) "
        f"is a WW2-era fighter. It is unlocked after purchasing it for $700,000 in the Plane Hangar at Rebirth 7. "
        f"It has a seating capacity of 1, 1 hull, 1 weapon, 1 engine, and the utility 'Zoom In'."
    )
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_general_info",
        values=get_embedding(general_info_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "general_info",
            "price": 700000,
            "unlock_method": "Purchase",
            "unlock_location": "Plane Hangar",
            "unlock_rebirth_level": 7,
            "seating_capacity": 1,
            "hulls": 1,
            "weapons_count": 1,
            "engines_count": 1,
            "utility": ["Zoom In"]
        },
        text_content=general_info_text
    ))

    # --- Chunk 2: Full Overview Text ---
    full_overview_text = """The P-51 Mustang is regarded as one of the weakest and most vulnerable vehicles in the entire game due to having no flares. Since it's a plane, it does not have the luxury of having the maneuverability of helicopters to linger in their flares, avoid lock-on missiles, and counter anti-air vehicles such as the Pantsir S1 and Patriot AA, which are extremely effective against the P-51 and other planes. The P-51 Mustang is equipped with four 20mm cannons mounted to its wings, which function similarly to a machine gun due to their lower individual fire rate when compared to the rotary cannons mounted to other planes such as the F-4 Phantom, F-14 Tomcat, and F-16 Falcon to name a few. The P-51's 20mm cannons combined, however, deal significant damage-per-shot and are able to fire over a larger area with each cannon shooting in an alternating pattern to maximise fire rate and coverage. The P-51 Mustang also comes with a single .50 caliber machine gun mounted underneath the nose, which, although by itself has a limited effectiveness, when combined with the more powerful 20mm cannons, the .50 caliber provides extra attrition damage against enemy vehicles and exposed infantry. Much like its real-life counterpart, the P-51 Mustang serves as an effective plane for Close Air Support (CAS) roles. Its powerful Area of Effect (AoE) damage makes the platform well suited for engaging lightly armored vehicles and exposed infantry whilst also being able to finish off severely damaged tanks in specific circumstances.
Evaluating this plane, if used correctly, the P-51 Mustang is a cheap, effective CAS aircraft but falters in other roles due to the divide between its skill requirements and effectiveness against other, more advanced planes.
Stats
Firepower
Armament
Damage Per Shot (Non-Upgraded)
Damage Per Shot (Tier 1)
Damage Per Shot (Tier 2)
Damage Per Shot (Tier 3)
20mm Cannons
[TBA]
[TBA]
[TBA]
[TBA]
.50 Caliber Machine Gun
[TBA]
[TBA]
[TBA]
[TBA]
Speed
Speed (Non-Upgraded)
Speed (Tier 1)
Speed (Tier 2)
Speed (Tier 3)
205 MPH
[TBA] MPH
[TBA] MPH
266 MPH
Health
Health (Non-Upgraded)
Health (Tier 1)
Health (Tier 2)
Health (Tier 3)
650 HP
715 HP
780 HP
845 HP
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
        f"The {item_name} is considered one of the weakest vehicles due to no flares, "
        f"vulnerable to lock-on missiles and anti-air. "
        f"However, it excels in Close Air Support (CAS) roles with powerful AoE damage "
        f"against lightly armored vehicles and infantry. "
        f"It's a cheap, effective CAS aircraft but struggles in other roles."
    )
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_overview_concise",
        values=get_embedding(concise_overview_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "overview_summary",
            "strengths": ["Close Air Support (CAS)", "Area of Effect (AoE) damage", "engaging lightly armored vehicles and exposed infantry", "finishing off severely damaged tanks", "cheap"],
            "weaknesses": ["no flares", "vulnerable to lock-on missiles", "vulnerable to anti-air"],
            "role": "Close Air Support (CAS)"
        },
        text_content=concise_overview_text
    ))

    # --- Chunk 4: Armament - 20mm Cannons ---
    cannons_armament_text = "Equipped with four 20mm cannons mounted to its wings, which function similarly to a machine gun due to their lower individual fire rate compared to rotary cannons. Combined, they deal significant damage-per-shot and fire over a larger area with alternating patterns."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_20mm_cannons",
        values=get_embedding(cannons_armament_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": "20mm Cannons",
            "count": 4,
            "mount_location": "wings",
            "function_analogy": "machine gun",
            "damage_characteristic": "significant damage-per-shot (combined)"
        },
        text_content=cannons_armament_text
    ))

    # --- Chunk 5: Armament - .50 Caliber Machine Gun ---
    mg_armament_text = "Comes with a single .50 caliber machine gun mounted underneath the nose. Limited effectiveness by itself, but provides extra attrition damage when combined with 20mm cannons."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_50cal_mg",
        values=get_embedding(mg_armament_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": ".50 Caliber Machine Gun",
            "count": 1,
            "mount_location": "underneath the nose",
            "effectiveness_alone": "limited",
            "combined_effectiveness": "extra attrition damage"
        },
        text_content=mg_armament_text
    ))

    # --- Chunk 6: Stats - Speed ---
    speed_text = "Speed (Minimum): 205 MPH, Speed (Maximum): 266 MPH. Tier 1 and Tier 2 speeds are [TBA]."
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
                "Tier 1": None, # [TBA] represented as None
                "Tier 2": None, # [TBA] represented as None
                "Tier 3": 266
            }
        },
        text_content=speed_text
    ))

    # --- Chunk 7: Stats - Health ---
    health_text = "Health (Minimum): 650 HP, Health (Maximum): 845 HP. Tier 1: 715 HP, Tier 2: 780 HP."
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
                "Non-Upgraded": 650,
                "Tier 1": 715,
                "Tier 2": 780,
                "Tier 3": 845
            }
        },
        text_content=health_text
    ))

    # --- Chunk 8: Category Information for this Plane ---
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


# --- Main Execution Block for P-51 ---
if __name__ == "__main__":
    # Ensure the output file is empty or doesn't exist to start fresh for P-51
    p51_output_jsonl = "pinecone_p51_vectors.jsonl"
    p51_output_pretty_json = "pinecone_p51_vectors_pretty.json"

    if os.path.exists(p51_output_jsonl):
        os.remove(p51_output_jsonl)
        print(f"Removed existing {p51_output_jsonl} to start fresh.")

    # --- P-51 Mustang Webpage Content (hardcoded) ---
    p51_webpage_content = r"""The North American P-51 Mustang (referred to as simply the 'P-51 Mustang' in War Tycoon) is a WW2-era fighter. It is unlocked after purchasing it for $700,000 in the Plane Hangar at Rebirth 7. It has a seating capacity of 1, 1 hull, 1 weapon, 1 engine, and the utility 'Zoom In'.
The P-51 Mustang is regarded as one of the weakest and most vulnerable vehicles in the entire game due to having no flares. Since it's a plane, it does not have the luxury of having the maneuverability of helicopters to linger in their flares, avoid lock-on missiles, and counter anti-air vehicles such as the Pantsir S1 and Patriot AA, which are extremely effective against the P-51 and other planes. The P-51 Mustang is equipped with four 20mm cannons mounted to its wings, which function similarly to a machine gun due to their lower individual fire rate when compared to the rotary cannons mounted to other planes such as the F-4 Phantom, F-14 Tomcat, and F-16 Falcon to name a few. The P-51's 20mm cannons combined, however, deal significant damage-per-shot and are able to fire over a larger area with each cannon shooting in an alternating pattern to maximise fire rate and coverage. The P-51 Mustang also comes with a single .50 caliber machine gun mounted underneath the nose, which, although by itself has a limited effectiveness, when combined with the more powerful 20mm cannons, the .50 caliber provides extra attrition damage against enemy vehicles and exposed infantry. Much like its real-life counterpart, the P-51 Mustang serves as an effective plane for Close Air Support (CAS) roles. Its powerful Area of Effect (AoE) damage makes the platform well suited for engaging lightly armored vehicles and exposed infantry whilst also being able to finish off severely damaged tanks in specific circumstances.
Evaluating this plane, if used correctly, the P-51 Mustang is a cheap, effective CAS aircraft but falters in other roles due to the divide between its skill requirements and effectiveness against other, more advanced planes.
Stats
Firepower
Armament
Damage Per Shot (Non-Upgraded)
Damage Per Shot (Tier 1)
Damage Per Shot (Tier 2)
Damage Per Shot (Tier 3)
20mm Cannons
[TBA]
[TBA]
[TBA]
[TBA]
.50 Caliber Machine Gun
[TBA]
[TBA]
[TBA]
[TBA]
Speed
Speed (Non-Upgraded)
Speed (Tier 1)
Speed (Tier 2)
Speed (Tier 3)
205 MPH
[TBA] MPH
[TBA] MPH
266 MPH
Health
Health (Non-Upgraded)
Health (Tier 1)
Health (Tier 2)
Health (Tier 3)
650 HP
715 HP
780 HP
845 HP
"""
    print("\n--- Processing P-51 Mustang ---")

    # Apply cleaning steps directly to the hardcoded string
    cleaned_p51_content = _scrub_webpage_content_chars(p51_webpage_content)
    cleaned_p51_content = re.sub(r'\s*\d+\s*[\u200B-\u200D\uFEFF]?\s*\uD83D[\uDCAD\uDCE4\uDCAC\uDD8E\uDD81-\uDD8E\u200B-\u200D\uFEFF]?[^\n]*?SIGN IN TO EDIT.*?[\u200B-\u200D\uFEFF]?\u22EE\s*', '', cleaned_p51_content, flags=re.DOTALL)
    cleaned_p51_content = re.sub(r'SIGN IN TO EDIT.*', '', cleaned_p51_content)
    cleaned_p51_content = re.sub(r'Contents\s*\[hide\].*?(?=(Overview|\d+\s*Overview|\d+\s*Stats|\d+\s*Firepower|\d+\s*Speed|\d+\s*Health))', '', cleaned_p51_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned_p51_content = re.sub(r'^\s*\d+(\.\d+)?\s*$', '', cleaned_p51_content, flags=re.MULTILINE)
    cleaned_p51_content = re.sub(r'\n{3,}', '\n\n', cleaned_p51_content)
    cleaned_p51_content = cleaned_p51_content.strip()

    p51_vectors = parse_p51_webpage_content(cleaned_p51_content)
    print(f"Generated {len(p51_vectors)} vectors for P-51 Mustang.")
    save_vectors_to_jsonl(p51_vectors, filepath=p51_output_jsonl) # Save to a specific file
    write_pretty_json_output(input_jsonl_filepath=p51_output_jsonl, output_json_filepath=p51_output_pretty_json)

    print(f"\nProcessing complete. Check '{p51_output_jsonl}' (for Pinecone) and '{p51_output_pretty_json}' (for review) files.")