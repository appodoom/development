import re
from utils import load_json, save_json
from defaults import VOCAB_JSON_PATH, TOKENS_JSON_PATH, CORPUS_JSON_PATH


class TrieNode:
    __slots__ = ("children", "token_id")

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
    atom_pattern = re.compile(r"[A-Za-z]+_\d+_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_\d+")
    vocab_list = load_json(json_file_path=VOCAB_JSON_PATH)
    trie = build_trie(vocab_list, atom_pattern)
    corpus_data = load_json(json_file_path=CORPUS_JSON_PATH)
    print("Tokenizing corpus data...")
    merged_tokens = tokenize_atoms(corpus_data, trie, vocab_list)
    save_json(output_json_path=TOKENS_JSON_PATH, data=merged_tokens)
    print("Tokenization Done.")


if __name__ == "__main__":
    main()
