import json

def save_eval_data(qa_data, filename="eval_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
