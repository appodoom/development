"""
This code merges corpuses of multiple wav files (old_samples) to do bpe on the merged corpus.
"""


import os

input_folder = "./demo_with_amp/corpus_with_amp"
output_file = "./demo_with_amp/corpus_with_amp/merged_enhanced.txt"


txt_files = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith(".txt")
])

all_tokens = []

for i,fname in enumerate(txt_files):
    file_path = os.path.join(input_folder, fname)
    with open(file_path, "r") as infile:
        lines = infile.readlines()
        tokens = [line.strip() for line in lines if line.strip()]
        all_tokens.extend(tokens)
    all_tokens.append('<EOF>')

# Write all tokens as a single line separated by spaces
with open(output_file, "w") as outfile:
    outfile.write(" ".join(all_tokens))

print(f"Done! Merged {len(txt_files)} files into {output_file} with {len(all_tokens)} tokens.")
