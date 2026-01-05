import regex as re
from collections import Counter
from typing import Optional, List, Tuple, Dict, Set

class BPETokenizer:
    def __init__(self, vocab_size: int = 500, input_path: str = "", special_tokens: list[str] | None = None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] - tokens that should not be split/merged
        '''
        self.vocab_size = vocab_size - len(special_tokens) if special_tokens else vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.merges = []

        self.input_path = input_path

        self.special_tokens = special_tokens or []
    
    def chunk_text(self) -> dict[tuple[int, ...], int]:
        """Pre-tokenize the corpus into regex chunks, split on special tokens first."""

        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:
            delim = "|".join(re.escape(st) for st in self.special_tokens)
            segments = re.split(delim, text)
        else:
            segments = [text]

        freq_table: dict[tuple[int, ...], int] = {}

        for seg in segments:
            for m in re.finditer(pattern, seg):
                chunk = m.group(0)
                if chunk:
                    ids = tuple(chunk.encode("utf-8"))
                    freq_table[ids] = freq_table.get(ids, 0) + 1

        return freq_table

    
    def iter_bpe(self, freq_table: dict[tuple[int, ...], int], vocab_id: int):

        count_table = {}

        for tokens, freq in freq_table.items():
            n = len(tokens)
            if n < 2:
                continue
            for i in range(n-1):
                pair = (tokens[i], tokens[i+1])
                count_table[pair] = count_table.get(pair, 0) + freq

        #pair = max(count_table.items(), key=lambda kv: (kv[1], kv[0]))[0]
        pair = max(
            count_table.items(),
            key=lambda kv: (kv[1], self.vocab[kv[0][0]], self.vocab[kv[0][1]]),
        )[0]

        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))

        word_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab[vocab_id] = word_bytes

        #word_str = word_bytes.decode("utf-8", errors="replace")
        #self.lookup_table[word_str] = vocab_id

        new_freq_table = {}
        pair0, pair1 = pair

        for tokens, freq in freq_table.items():
            n = len(tokens)
            if n < 2:
                new_freq_table[tokens] = new_freq_table.get(tokens, 0) + freq
                continue

            i = 0
            out = []
            append = out.append  # 本地绑定，少一次属性查找

            while i < n - 1:
                if tokens[i] == pair0 and tokens[i + 1] == pair1:
                    append(vocab_id)
                    i += 2
                else:
                    append(tokens[i])
                    i += 1

            if i == n - 1:
                append(tokens[-1])
            out = tuple(out)

            new_freq_table[out] = new_freq_table.get(out, 0) + freq

        return new_freq_table


    def train(self) -> None:
        freq_table = self.chunk_text()

        for i in range(256, self.vocab_size):
            freq_table = self.iter_bpe(freq_table,i)
        
        for i, tok in enumerate(self.special_tokens):
            self.vocab[self.vocab_size + i] = tok.encode("utf-8")

        return self.vocab, self.merges

if __name__ == "__main__":
    with open("../text.txt", "r", encoding="utf-8") as f:
        text = f.read()
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)

    vocab_size = 260
    BPE = BPETokenizer(vocab_size, text)
    vocab, merges = BPE.train()

