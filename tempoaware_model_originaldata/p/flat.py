import json

# Input JSON file (assumes it's a flat list like ["kick", "snare", "hihat", ...])
json_path = "corpus_results.json"
txt_path = "corpus_results.txt"

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Write to TXT
with open(txt_path, "w") as f:
    for item in data:
        f.write(f"{item}\n")

print(f"✅ Converted {json_path} to {txt_path}")
