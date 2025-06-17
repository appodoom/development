import re
import json
from pathlib import Path

class TrieNode:
    __slots__ = ('children', 'token_id')

    def __init__(self):
        self.children = {}
        self.token_id = None


def build_trie(vocab_list, atom_pattern):
    """
    Build a trie where each merge token (sequence of atomic tokens) from vocab_list is inserted.
    Atomic tokens are extracted via atom_pattern.findall on each merge string.
    """
    root = TrieNode()
    for idx, merge in enumerate(vocab_list):
        node = root
        atoms = atom_pattern.findall(merge)
        if not atoms:
            raise ValueError(f"Merge token '{merge}' contains no atomic tokens.")
        for atom in atoms:
            node = node.children.setdefault(atom, TrieNode())
        node.token_id = idx
    return root


def tokenize_atoms(atoms, trie, vocab_list):
    """
    Tokenize the list of atomic tokens into merge-tokens using longest-prefix matching.
    Raises ValueError if no match is found at a position.
    Returns list of vocab_list entries (merge strings).
    """
    output = []
    i = 0
    n = len(atoms)
    while i < n:
        node = trie
        last_match = None
        last_len = 0
        j = i
        while j < n and atoms[j] in node.children:
            node = node.children[atoms[j]]
            j += 1
            if node.token_id is not None:
                last_match = node.token_id
                last_len = j - i
        if last_match is None:
            raise ValueError(f"Unknown atomic token '{atoms[i]}' at position {i}")
        output.append(vocab_list[last_match])
        i += last_len
    return output


def main():
    atom_pattern = re.compile(r'[A-Za-z]+_[0-9]+(?:\.[0-9]+)?')

    # Load vocabulary merges
    vocab_path = Path('vocab_no_amp_2000.txt')
    vocab_list = [line.strip() for line in vocab_path.read_text('utf-8').splitlines() if line.strip()]

    # Build trie for merge tokens
    trie = build_trie(vocab_list, atom_pattern)

    # Prepare output directory
    tokens_dir = Path('tokens_no_amp')
    tokens_dir.mkdir(exist_ok=True)

    # Process each corpus file
    corpus_dir = Path('corpus_no_amp')
    for corpus_file in sorted(corpus_dir.glob('old*.txt')):
        print(f"Tokenizing {corpus_file.name}...")
        # Read raw text and split into atomic tokens by whitespace
        atoms = corpus_file.read_text('utf-8').strip().split()
        merged_tokens = tokenize_atoms(atoms, trie, vocab_list)
        # Write out the merge-token sequence
        out_path = tokens_dir / f"tokens_{corpus_file.stem}.json"
        out_path.write_text(json.dumps(merged_tokens, ensure_ascii=False, indent=2), encoding='utf-8')

    print("All corpus files tokenized.")

if __name__ == '__main__':
    main()
