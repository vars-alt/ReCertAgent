from datasets import load_dataset

def load_notinject(limit=None):
    dataset = load_dataset("leolee99/NotInject")
    rows = []
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            rows.append({
                "source_id": f"{split_name}:{index}",
                "prompt": row["prompt"],
                "category": row.get("category", ""),
                "trigger_words": row.get("word_list", []),
            })
            if limit and len(rows) >= limit:
                return rows
    if not rows:
        raise RuntimeError("NotInject returned no examples.")
    return rows
