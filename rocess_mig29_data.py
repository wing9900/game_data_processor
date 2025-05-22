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

# --- Parsing Function for MiG-29 Fulcrum ---
def parse_mig29_webpage_content(webpage_text: str) -> List[PineconeVector]:
    """
    Parses the cleaned text content of a MiG-29 Fulcrum webpage
    into structured chunks for Pinecone.

    Args:
        webpage_text: The pre-cleaned content of the MiG-29 webpage.

    Returns:
        A list of dictionaries, each representing a Pinecone vector.
    """
    processed_vectors: List[PineconeVector] = []
    item_name = "MiG-29 Fulcrum"
    entity_type = "aircraft"

    # --- Chunk 1: General Information ---
    # Extracting from the initial paragraph and General Information table
    general_info_match = re.search(
        r"The Mikoyan-Gurevich MiG-29 Fulcrum is a Soviet-Era multi-role fighter jet used primarily by the Russian Air Force\."
        r"\s*It is unlocked after completing Operation Aerial Ace and can be found in the Plane Hangar after reaching Rebirth 7\."
        r".*?Price\s*\$([0-9,]+).*?" # Price
        r"Speed \(Minimum\)(.*?)(?:MPH|\s*\[TBA\]\s*MPH)Speed \(Maximum\)(.*?)(?:MPH|\s*\[TBA\]\s*MPH)" # Speeds
        r"Health \(Minimum\)(.*?)(?:HP)Health \(Maximum\)(.*?)(?:HP)" # Health
        r"Armament\s*-\s*(.+?);\s*-\s*(.+?)\." # Armament summary
        r"\s*Utility\s*-\s*(.+?);\s*-\s*(.+?);\s*-\s*(.+?)" # Utility summary
        r"\s*Seating Capacity\s*(\d+)" # Seating
        r"\s*Hulls\s*(\d+)\s*Weapons\s*(\d+)\s*Engines\s*(\d+)", # Hull, Weapons, Engines
        webpage_text, re.DOTALL
    )

    general_info_text_raw = (
        f"The Mikoyan-Gurevich MiG-29 Fulcrum is a Soviet-Era multi-role fighter jet used primarily by the Russian Air Force. "
        f"It is unlocked after completing Operation Aerial Ace and can be found in the Plane Hangar after reaching Rebirth 7. "
        f"It has a price of $850,000, seating capacity of 2, 1 hull, 2 weapons, 1 engine. "
        f"Its armament includes 1x 30mm Autocannon and 6x Air-to-Air Missiles. "
        f"Utilities include Flares, 2x Ejection Seats, and Zoom In."
    )

    price_str = general_info_match.group(1).replace(",", "") if general_info_match else "850000"
    min_speed_val = general_info_match.group(2).strip() if general_info_match else "275"
    max_speed_val = general_info_match.group(3).strip() if general_info_match else "[TBA]"
    min_health_val = general_info_match.group(4).strip() if general_info_match else "680"
    max_health_val = general_info_match.group(5).strip() if general_info_match else "884"
    armament1 = general_info_match.group(6).strip() if general_info_match else "1x 30mm Autocannon"
    armament2 = general_info_match.group(7).strip() if general_info_match else "6x Air-to-Air Missiles"
    utility1 = general_info_match.group(8).strip() if general_info_match else "Flares"
    utility2 = general_info_match.group(9).strip() if general_info_match else "2x Ejection Seats"
    utility3 = general_info_match.group(10).strip() if general_info_match else "Zoom In"
    seating = general_info_match.group(11).strip() if general_info_match else "2"
    hulls = general_info_match.group(12).strip() if general_info_match else "1"
    weapons = general_info_match.group(13).strip() if general_info_match else "2"
    engines = general_info_match.group(14).strip() if general_info_match else "1"

    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_general_info",
        values=get_embedding(general_info_text_raw),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "general_info",
            "price": int(price_str) if price_str.isdigit() else None,
            "unlock_method": "Operation Aerial Ace",
            "unlock_details": "Kill 30 enemy players with any plane",
            "hangar_access_rebirth_level": 7,
            "seating_capacity": int(seating),
            "hulls": int(hulls),
            "weapons_count": int(weapons),
            "engines_count": int(engines),
            "utility": [utility1, utility2, utility3],
            "initial_armament_summary": [armament1, armament2],
            "speed_min_display": min_speed_val,
            "speed_max_display": max_speed_val,
            "health_min_display": min_health_val,
            "health_max_display": max_health_val,
        },
        text_content=general_info_text_raw
    ))

    # --- Chunk 2: Full Overview Text ---
    # Extract the main overview section, excluding "General Information" and "Stats" blocks
    overview_text_match = re.search(
        r"Overview\nMiG-29 Fulcrum\n.*?General Information.*?The Mikoyan-Gurevich MiG-29 Fulcrum is known for its role as an early-game fighter jet within War Tycoon\.(.*?)(?=History\n|Stats\n)",
        webpage_text, re.DOTALL
    )
    full_overview_text = overview_text_match.group(1).strip() if overview_text_match else ""

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
        f"The {item_name} is an early-game multi-role fighter jet known for its relatively simple unlock requirements (Operation Aerial Ace) "
        f"and strong combat capabilities despite being an early-game option. "
        f"It features a potent 30mm autocannon effective for base shield destruction and air-to-ground engagements, "
        f"and 6 air-to-air missiles which can be used to bait enemy flares. "
        f"Unlike the P-51, it has flares, making it less vulnerable to lock-on missiles. "
        f"While not as agile as some modern jets at higher speeds, it excels in close-range dogfights at slower speeds. "
        f"It's a significant upgrade from the P-51 Mustang."
    )
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_overview_concise",
        values=get_embedding(concise_overview_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "overview_summary",
            "strengths": [
                "Early-game fighter", "Simple unlock", "Potent 30mm autocannon",
                "High damage-per-shot (autocannon)", "Effective for base shield destruction",
                "Good for air-to-ground", "6 air-to-air missiles", "Can bait enemy flares",
                "Has flares (unlike P-51)", "Smaller turn radius at slower speeds",
                "Good for close-range dogfights", "Significant upgrade over P-51"
            ],
            "weaknesses": [
                "Lower DPS (autocannon) compared to 20mm rotary cannons",
                "Smaller damage-per-hit (missiles)", "Requires more missiles to cripple/destroy",
                "Jets cannot linger in flares like helicopters", "More vulnerable to lock-on missiles (Stingers, Pantsir S1)",
                "Not as easy to handle as F-14 Tomcat", "Larger turn radius at higher speeds",
                "May falter against other, more modern planes"
            ],
            "role": "Early-game multi-role fighter",
            "playstyle_notes": "Use autocannon for ground/shields, missiles for air-to-air (including baiting flares), leverage slow-speed maneuverability in dogfights."
        },
        text_content=concise_overview_text
    ))

    # --- Chunk 4: Armament - 30mm Autocannon ---
    autocannon_text = "The MiG-29 possesses a potent 30mm rotary cannon mounted to the right of the cockpit's underside near the nose. This rotary cannon has a notably lower DPS (Damage Per Second) when compared to the likes of the 20mm rotary cannons mounted on the F-4 Phantom, F-14 Tomcat, and F-16 Falcon. As a direct trade-off to this lowered fire rate, the MiG-29's 30mm rotary cannon has a higher damage-per-shot, allowing it to fulfill roles in base shield destruction and air-to-ground engagements."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_30mm_autocannon",
        values=get_embedding(autocannon_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": "30mm Autocannon",
            "count": 1,
            "mount_location": "right of cockpit's underside near nose",
            "dps_comparison": "lower DPS than 20mm rotary cannons",
            "damage_characteristic": "higher damage-per-shot",
            "roles": ["base shield destruction", "air-to-ground engagements"]
        },
        text_content=autocannon_text
    ))

    # --- Chunk 5: Armament - Air-to-Air Missiles ---
    missiles_text = "The MiG-29 also possesses a compliment of 6 air-to-air missiles for use against enemy aerial targets, however, the trade-off to the fighter's increased munitions payload is the smaller damage-per-hit of each missile. This means that the pilot of the MiG-29 needs to fire more missiles against an enemy target to severely cripple or destroy it, however, having access to so many missiles allows pilots to \"bait\" enemy aircraft into dispensing their flares prematurely, allowing MiG-29 pilots to swiftly launch the remainder of their payload and deal significant damage."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_armament_air_to_air_missiles",
        values=get_embedding(missiles_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "armament",
            "weapon_type": "Air-to-Air Missiles",
            "count": 6,
            "damage_characteristic": "smaller damage-per-hit",
            "usage_strategy": "can bait enemy aircraft into dispensing flares prematurely"
        },
        text_content=missiles_text
    ))

    # --- Chunk 6: Stats - Speed ---
    speed_text = "Speed (Non-Upgraded): 275 MPH, Speed (Tier 1): [TBA] MPH, Speed (Tier 2): [TBA] MPH, Speed (Tier 3): [TBA] MPH."
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
                "Non-Upgraded": 275,
                "Tier 1": None, # [TBA]
                "Tier 2": None, # [TBA]
                "Tier 3": None  # [TBA]
            }
        },
        text_content=speed_text
    ))

    # --- Chunk 7: Stats - Health ---
    health_text = "Health (Non-Upgraded): 680 HP, Health (Tier 1): 748 HP, Health (Tier 2): 816 HP, Health (Tier 3): 884 HP."
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
                "Non-Upgraded": 680,
                "Tier 1": 748,
                "Tier 2": 816,
                "Tier 3": 884
            }
        },
        text_content=health_text
    ))

    # --- Chunk 8: History ---
    # Extracting the entire History section
    history_text_match = re.search(
        r"History\s*(.*)",
        webpage_text, re.DOTALL
    )
    # The previous regex might capture too much. Let's try to refine the history capture
    # from "During the Vietnam war" until the next "Stats" or end of document
    history_content_match = re.search(
        r"(During the Vietnam war, it was clear to the USAF.*?)(?=Stats\nFirepower|Speed\n|Health\n|$)",
        webpage_text, re.DOTALL
    )
    history_text = history_content_match.group(1).strip() if history_content_match else ""

    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_history",
        values=get_embedding(history_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "history",
            "section_title": "History",
            "period": ["Vietnam War", "Cold War", "1960s", "1970s", "1980s"], # Example metadata
            "key_events": ["F-X program", "PFI program", "LPFI program", "Maiden flight 1977"]
        },
        text_content=history_text
    ))

    # --- Chunk 9: Category Information for this Plane ---
    category_text = f"The {item_name} is classified under the 'Fighter Jets' category of planes."
    processed_vectors.append(PineconeVector(
        id=f"{item_name.lower().replace(' ', '_')}_category_membership",
        values=get_embedding(category_text),
        metadata={
            "entity_type": entity_type,
            "item_name": item_name,
            "info_type": "category_membership",
            "aircraft_category": "Fighter Jets" # Updated to "Fighter Jets"
        },
        text_content=category_text
    ))

    return processed_vectors


# --- Main Execution Block ---
if __name__ == "__main__":
    mig29_output_jsonl = "pinecone_mig29_vectors.jsonl"
    mig29_output_pretty_json = "pinecone_mig29_vectors_pretty.json"

    if os.path.exists(mig29_output_jsonl):
        os.remove(mig29_output_jsonl)
        print(f"Removed existing {mig29_output_jsonl} to start fresh.")

    # Raw content for MiG-29 Fulcrum
    mig29_webpage_content = r"""
9


SIGN IN TO EDIT

The Mikoyan-Gurevich MiG-29 Fulcrum is a Soviet-Era multi-role fighter jet used primarily by the Russian Air Force.
It is unlocked after completing Operation Aerial Ace and can be found in the Plane Hangar after reaching Rebirth 7.

Contents

1Overview
2History
3Stats
3.1Firepower
3.2Speed
3.3Health
Overview
MiG-29 Fulcrum

The in-game render for the MiG-29 Fulcrum.
General Information
Price
$850,000
(Requires Operation Aerial Ace to be completed.)
Speed (Minimum)Speed (Maximum)275 MPH[TBA] MPH
Health (Minimum)Health (Maximum)680 HP884 HP
Armament
- 1x 30mm Autocannon;
- 6x Air-to-Air Missiles.
Utility
- Flares;
- 2x Ejection Seats;
- Zoom In
Seating Capacity
2
Vehicle Parts Cost
HullsWeaponsEngines121

The Mikoyan-Gurevich MiG-29 Fulcrum is known for its role as an early-game fighter jet within War Tycoon. Known for its easy Operation, the requirements to unlock the MiG-29 are relatively simple, requiring the player to kill 30 enemy players with any plane.
The MiG-29 possesses a potent 30mm rotary cannon mounted to the right of the cockpit's underside near the nose. This rotary cannon has a notably lower DPS (Damage Per Second) when compared to the likes of the 20mm rotary cannons mounted on the F-4 Phantom, F-14 Tomcat, and F-16 Falcon. As a direct trade-off to this lowered fire rate, the MiG-29's 30mm rotary cannon has a higher damage-per-shot, allowing it to fulfill roles in base shield destruction and air-to-ground engagements.
The MiG-29 also possesses a compliment of 6 air-to-air missiles for use against enemy aerial targets, however, the trade-off to the fighter's increased munitions payload is the smaller damage-per-hit of each missile. This means that the pilot of the MiG-29 needs to fire more missiles against an enemy target to severely cripple or destroy it, however, having access to so many missiles allows pilots to "bait" enemy aircraft into dispensing their flares prematurely, allowing MiG-29 pilots to swiftly launch the remainder of their payload and deal significant damage.
Unlike the P-51 Mustang, the MiG-29 has flares, rendering the P-51 Mustang outclassed compared to this jet. However, jets do not have the luxury of staying in their flares like Helicopters, which makes them more vulnerable to lock-on missiles originating from Stingers, the Pantsir S1, and helicopters. The MiG-29 is not as easy to handle compared to the F-14 Tomcat and has a larger turn radius when compared to its contemporaries at higher speed, however, the MiG-29 possesses a smaller turn radius when traveling at a slower speed, allowing it to combat enemy air targets in close range.
In conclusion, the MiG-29 Fulcrum serves as a significant upgrade compared to the P-51 Mustang, serving as a good early-game fighter jet. When compared to other, more modern planes, the MiG-29 may falter.
History
During the Vietnam war, it was clear to the USAF that low altitude supersonic fighter bombers, like the F-105 Thunderchief and F-104 Starfighter, were extremely vulnerable to the older Soviet models, such as the MiG-17's and more advanced ones such as the MiG-21 were far more more maneuverable. To help regain air superiority over Vietnam, the U.S. employed the F-4 Phantom, while the USSR used the MiG-23 in response. In the late 1960s the USAF started the "F-X" program to build a fighter that would have total air superiority, the following aircraft would become the McDonnell Douglas F-15 Eagle soon after being ordered for production in 1969. During one of the most tense moments of the Cold War, the USSR needed a response or else they would lag behind the United States in technological developments, hence the development of an air superiority fighter was necessary. In the same year the Soviet General Staff would issue the requirement for a Perspektivnyy Frontovoy Istrebitel (PFI "Advanced Frontline Fighter"). The list of demands were ambitious, calling for long range, be able to land almost anywhere such as austere runways, exceptional maneuverability, Mach 2+ speed, and the ability to carry almost everything.
By 1971, a different type of fighter was needed for the Soviet Union. The PFI program was slightly changed with the Perspektivnyy Lyogkiy Frontovoy Istrebitel (LPFI, or "Advanced Lightweight Tactical Fighter") program. The Soviets planned to have a fleet of 33% PFI planes and 66% LPFI planes. This decision aligned closely with the USAF's decision of the Lightweight Fighter program. The role of designing a PFI fighter went to Sukhoi, resulting in the well known Su-27 "Flanker", while the LPFI was assigned to Mikoyan. Production of the MiG-29 would start in 1974, taking its maiden flight on 6 October 1977. The MiG-29 would soon replace the older MiG-23 throughout the 1980s alongside the Su-27.
The aircraft would be given the NATO Reporting name "Fulcrum-A". Soon the aircraft would be widely exported to many of the Warsaw pact countries in downgraded versions. Today while not as impressive as other fighters the Fulcrum does its job extremely well with many countries still operating today including former Warsaw pact members and Russia.
Stats
Firepower
Armament Stats CollapseArmamentDamage Per Shot (Non-Upgraded)Damage Per Shot (Tier 1)Damage Per Shot (Tier 2)Damage Per Shot (Tier 3)30mm Autocannon[TBA][TBA][TBA][TBA]Air-to-Air Missiles[TBA][TBA][TBA][TBA]
Speed
Speed Stats CollapseSpeed (Non-Upgraded)Speed (Tier 1)Speed (Tier 2)Speed (Tier 3)275 MPH[TBA] MPH[TBA] MPH[TBA] MPH
Health
Health Stats CollapseHealth (Non-Upgraded)Health (Tier 1)Health (Tier 2)Health (Tier 3)680 HP748 HP816 HP884 HP
"""
    print("\n--- Processing MiG-29 Fulcrum ---")

    # Apply cleaning steps directly to the hardcoded string content.
    cleaned_mig29_content = _scrub_webpage_content_chars(mig29_webpage_content)
    cleaned_mig29_content = re.sub(r'\s*\d+\s*[\u200B-\u200D\uFEFF]?\s*\uD83D[\uDCAD\uDCE4\uDCAC\uDD8E\uDD81-\uDD8E\u200B-\u200D\uFEFF]?[^\n]*?SIGN IN TO EDIT.*?[\u200B-\u200D\uFEFF]?\u22EE\s*', '', cleaned_mig29_content, flags=re.DOTALL)
    cleaned_mig29_content = re.sub(r'SIGN IN TO EDIT.*', '', cleaned_mig29_content)
    # Specific cleanup for the "Contents" block which is slightly different here
    cleaned_mig29_content = re.sub(r'Contents\s*1Overview\s*2History\s*3Stats\s*3.1Firepower\s*3.2Speed\s*3.3Health', '', cleaned_mig29_content, flags=re.DOTALL)
    cleaned_mig29_content = re.sub(r'^\s*\d+(\.\d+)?\s*$', '', cleaned_mig29_content, flags=re.MULTILINE) # Remove standalone numbers
    cleaned_mig29_content = re.sub(r'Collapse', '', cleaned_mig29_content) # Remove "Collapse" text
    cleaned_mig29_content = re.sub(r'\n{3,}', '\n\n', cleaned_mig29_content) # Remove excessive blank lines
    cleaned_mig29_content = cleaned_mig29_content.strip()


    mig29_vectors = parse_mig29_webpage_content(cleaned_mig29_content)
    print(f"Generated {len(mig29_vectors)} vectors for MiG-29 Fulcrum.")
    save_vectors_to_jsonl(mig29_vectors, filepath=mig29_output_jsonl)
    write_pretty_json_output(input_jsonl_filepath=mig29_output_jsonl, output_json_filepath=mig29_output_pretty_json)

    print(f"\nProcessing complete. Check '{mig29_output_jsonl}' (for Pinecone) and '{mig29_output_pretty_json}' (for review) files.")