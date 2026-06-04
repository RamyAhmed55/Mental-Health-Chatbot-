import os
import json
import pandas as pd
from datasets import load_dataset

RECORDS_FILE = os.path.join(os.path.dirname(__file__), "qa_records.json")

# =====================================================================================================

def load_records() -> list[dict]:

    if os.path.exists(RECORDS_FILE):
        print("Loading records from file...")
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")
        return records


    print("Downloading dataset & building records...")
    dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    df  = pd.DataFrame(dataset).dropna(subset=["Context", "Response"])
    records = _build_records(df)


    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


    print(f"Built & saved {len(records)} records")
    return records

# =====================================================================================================

def _build_records(df) -> list[dict]:

    records = []
    for idx, row in df.iterrows():
        context  = str(row["Context"]).strip()
        response = str(row["Response"]).strip()

        if not context or not response:
            continue
        records.append({
            "id"      : idx,
            "context" : context,
            "response": response,})
        
    return records