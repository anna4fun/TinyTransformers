from typing import Tuple, Dict, List

# The simplest tokenizer, every tokenizer contains the rule of how to break words and sentences, and a pair of encode and decode
def build_tokenizers(text: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    # NOTE: char-level; includes whitespace and punctuation by design
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s: str, stoi: Dict[str,int]) -> List[int]:
    # Will KeyError on unseen chars; fine as long as you fit on the full corpus.
    return [stoi[ch] for ch in s]

def decode(ids: List[int], itos: Dict[int,str]) -> str:
    return ''.join(itos[i] for i in ids)
