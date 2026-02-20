# swim/agents/homogen/utils/column_mapper.py

import difflib

# Canonical mapping dict (one-way)
CANONICAL_FIELDS = {
    "lake_name": ["lake", "see", "wassername", "waterbody", "wasserkoerper"],
    "date": ["date", "datum", "mess_datum", "sample_date", "sampling_date"],
    "ph": ["ph_wert", "ph"],
    "temperature": ["wassertemperatur", "temperature", "temp", "temp_water"],
    "turbidity": ["turbidität", "turbidity", "trübung", "trubung"],
    "oxygen": ["sauerstoff", "oxygen", "dissolved_oxygen"],
    "nitrate": ["nitrat", "nitrate", "no3"],
    "conductivity": ["leitfähigkeit", "conductivity", "konduktivitaet"],
    "tss": ["tss", "total_suspended_solids", "suspended_matter"],
}

def map_columns_auto(input_columns: list[str]) -> dict:
    """
    Map arbitrary column names to canonical schema using fuzzy matching.
    
    Returns:
        Dict like: {'Messdatum': 'date', 'Trübung': 'turbidity', ...}
    """
    result = {}
    used_targets = set()

    for col in input_columns:
        best_match = None
        highest_score = 0

        for target_field, aliases in CANONICAL_FIELDS.items():
            if target_field in used_targets:
                continue
            all_possible = [target_field] + aliases
            match = difflib.get_close_matches(col.lower(), all_possible, n=1, cutoff=0.6)
            if match:
                best_match = target_field
                highest_score = 1.0
                break

        if best_match:
            result[col] = best_match
            used_targets.add(best_match)
        else:
            result[col] = col  # Leave unmapped

    return result