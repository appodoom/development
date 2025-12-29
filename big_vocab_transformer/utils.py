import json


def save_json(output_json_path: str, data):
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content
