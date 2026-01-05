import regex as re
from typing import Dict, Tuple, Iterable, Iterator, List
import json
from functools import lru_cache

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens else None

        self.pair_rank = self._initial_pair_rank()
        # Create reverse lookup for faster token ID retrieval
        self.token_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

    def _iter_pretokenize(self, text: str) -> list[bytes]:
        str_tokens = re.findall(PATTERN, text)
        byte_tokens = [s.encode("utf-8") for s in str_tokens]
        return byte_tokens
    
    def _initial_pair_rank(self) -> dict[tuple[bytes, bytes], int]:
        """Create a mapping from byte pairs to their merge rank (priority)."""
        pair_rank = {}
        for i, (token1, token2) in enumerate(self.merges):
            pair_rank[(token1, token2)] = i
        return pair_rank


    @lru_cache(maxsize=5000)
    def _encode_word(self, word: bytes) -> List[int]:
        """Encode a single word using BPE merges."""
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in word]

        if len(tokens) == 1:
            # Single byte, just return its vocab ID
            return [self._get_token_id(tokens[0])]

        # Iteratively merge pairs according to the learned merges
        while len(tokens) >= 2:
            # Find all adjacent pairs and their ranks
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

            # Find the pair with the lowest rank (earliest in merge sequence)
            min_rank = float('inf')
            min_idx = -1

            for i, pair in enumerate(pairs):
                rank = self.pair_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_idx = i

            # If no valid merge found, break
            if min_idx == -1 or min_rank == float('inf'):
                break

            # Merge the pair at min_idx
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i == min_idx:
                    # Merge tokens[i] and tokens[i+1]
                    merged = tokens[i] + tokens[i+1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # Convert tokens to vocab IDs
        return [self._get_token_id(token) for token in tokens]

    def _get_token_id(self, token: bytes) -> int:
        """Get the vocab ID for a token."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        raise ValueError(f"Token {token} not found in vocabulary")



    def _encode_text(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        if not text:
            return []

        ids_out = []

        # Pretokenize the text into words
        word_bytes_all = self._iter_pretokenize(text)
        # word_bytes_all = [b'hello', b'you', b'are', b'a', b'cat']

        for word_bytes in word_bytes_all:
            # Encode each word using BPE
            word_ids = self._encode_word(word_bytes)
            ids_out.extend(word_ids)

        return ids_out

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r") as f:
            self.vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            self.merges = [tuple(line.strip().split(" ")) for line in f]
        if special_tokens:
            self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if not text:
            return []

        if self.special_tokens:
            # Sort special tokens by length (longest first) to match longer tokens first
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            delim = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            segments = re.split(delim, text)
        else:
            segments = [text]

        out = []
        for seg in segments:
            if self.special_tokens and seg in self.special_tokens:
                # Find the vocab ID for this special token
                special_token_bytes = seg.encode("utf-8")
                token_id = self._get_token_id(special_token_bytes)
                out.append(token_id)
            elif seg:  # Only encode non-empty segments
                out.extend(self._encode_text(seg))
        return out


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle),
         return a generator that lazily yields token IDs.
         This is required for memory-efficient tokenization of large files that
          we cannot directly load into memory."""
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        if not ids:
            return ""

        # Convert token IDs to bytes
        tokens = []
        for id in ids:
            if id in self.vocab:
                tokens.append(self.vocab[id])
            else:
                raise ValueError(f"Token ID {id} not found in vocabulary")

        # Concatenate all tokens and decode to string
        byte_string = b"".join(tokens)
        return byte_string.decode("utf-8", errors="replace")

if __name__ == "__main__":
    text = 'hello you are a cat'
    str_tokens = re.findall(PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    print(byte_tokens)

    text = 'hello you are a cat <|endoftext|> do you like build a snowman?'
    st = "<|endoftext|>"
    delim = "|".join(re.escape(st) for st in ["<|endoftext|>", "<|startoftext|>"])
    #delim = re.escape(st)
    print(delim)
    segments = re.split(delim, text)
    print(segments)

    for seg in segments:
        for m in re.finditer(PATTERN, seg):
            part = m.group(0)
            print(part.encode("utf-8"))