import re

def sort_vocab_by_hit_count(vocab_path, output_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]

    def count_hits(token):
        # Count how many full hits are in the token
        # All hits are assumed to be of format: letter + "_" + number (e.g. b_21.33)
        return len(re.findall(r'[a-zA-Z]_\d+(?:\.\d+)?', token))

    # Sort tokens by hit count (descending)
    tokens_sorted = sorted(tokens, key=lambda tok: count_hits(tok), reverse=True)

    # Write to output
    with open(output_path, "w", encoding="utf-8") as f:
        for tok in tokens_sorted:
            f.write(f"{tok}\n")

    print(f"✅ Vocab sorted by number of hits saved to {output_path}")

# Run it
if __name__ == "__main__":
    sort_vocab_by_hit_count(
        vocab_path="vocab_music_sheet_250.txt",
        output_path="vocab_sorted_by_hits.txt"
    )
