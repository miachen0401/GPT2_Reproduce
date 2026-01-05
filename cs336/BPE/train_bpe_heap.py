import heapq
from collections import Counter
from typing import Optional, List, Tuple, Dict, Set
import regex as re
from concurrent.futures import ProcessPoolExecutor, as_completed

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab_size: int = 10_000, input_path: str = "", special_tokens: List[str] = None):
        self.vocab_size = vocab_size - len(special_tokens) if special_tokens else vocab_size
        self.input_path = input_path
        self.special_tokens = special_tokens

        self._delimiter = special_tokens[0] if special_tokens else None

        self.freq_table: Dict[Tuple[int,...], int] = {}
        self.pair_to_words: Dict[Tuple[int, int], Set[Tuple[int, ...]]] = {}
        self.pair_counts: Dict[Tuple[int, int], int] = {}
        self.maxheap: list[tuple[int, tuple[int, int]]] = []

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: List[Tuple[bytes, bytes]] = []

        self.count = 0
    
    def _bytes_to_tuple(self, freq_table: Counter[bytes:int]):
        for b, f in freq_table.items():
            if not b:
                continue
            self.freq_table[tuple(b)]=f
        return
    
    def _covert_tuple(self, pair: Tuple[int, int]):
        return tuple(-p for p in pair)

    def invert_bytes(self, b: bytes, l: int = 3) -> tuple[int, ...]:
        b = [255 - x for x in b]
        b.extend([255] * (l - len(b)))
        return tuple(b)

    def _heappush(self, pair: tuple[int, int], count: int):
        b0 = self.vocab[pair[0]]
        b1 = self.vocab[pair[1]]

        k0 = self.invert_bytes(b0)
        k1 = self.invert_bytes(b1)

        heapq.heappush(self.maxheap, (-count, k0, k1, pair))

    def _heappop(self):
        while True:
            neg_count, p0, p1, pair = heapq.heappop(self.maxheap)
            count = - neg_count
            if count == self.pair_counts.get(pair, 0):
                break
        self.count += 1
        if 95 < self.count < 120:
            tmp = self.vocab[pair[0]] + self.vocab[pair[1]]
            print(count, p0, p1, pair, tmp)
        return count, pair


    def chunk_text(self, chunk_size: int = 128 * 1024 * 1024, 
                    n_workers = 4):
        with open(self.input_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()

        freq_table = {}
        
        if self.special_tokens:
            delim = "|".join(re.escape(st) for st in self.special_tokens)
            segments = re.split(delim, chunk_text)
        else:
            segments = [chunk_text]
            
        for seg in segments:
            for m in re.finditer(PATTERN, seg):
                part = m.group(0)
                if not part:
                    continue
                freq_table[part.encode("utf-8")] = freq_table.get(part.encode("utf-8"), 0) + 1
        return freq_table
    

    def iter_BPE(self, vocab_id: int):
        count, pair = self._heappop()

        self.vocab[vocab_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))

        new_pair_counts: Dict[Tuple[int, int], int] = {}

        words_affected = self.pair_to_words[pair].copy()
        del self.pair_to_words[pair]


        for word in words_affected: # (109, 23, 25, 89, 111)
            if word not in self.freq_table:
                continue

            word_freq = self.freq_table[word]

            for t in range(len(word) - 1):
                old_pair = (word[t], word[t + 1])
                self.pair_counts[old_pair] -= word_freq

            del self.freq_table[word]

            new_word = []
            i = 0
            while i < len(word)-1:
                if word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(vocab_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word)-1:
                new_word.append(word[i])

            new_word = tuple(new_word)
            self.freq_table[new_word] = self.freq_table.get(new_word, 0) + word_freq

            for t in range(len(new_word)-1):
                new_pair = (new_word[t], new_word[t+1])
                new_pair_counts[new_pair] = new_pair_counts.get(new_pair, 0) + word_freq
                self.pair_to_words[new_pair] = self.pair_to_words.get(new_pair, set()) | {new_word}

        del self.pair_counts[pair]
        
        for k, v in new_pair_counts.items():
            self.pair_counts[k] = self.pair_counts.get(k, 0) + v
            self._heappush(k, self.pair_counts[k])

        return

    def train_BPE(self):
        freq = self.chunk_text()
        self._bytes_to_tuple(freq)

        for k, f in self.freq_table.items():
            for i in range(len(k)-1):
                pair = (k[i], k[i+1])
                self.pair_to_words[pair] = self.pair_to_words.get(pair, set()) | {k}
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + f

        for pair, counts in self.pair_counts.items():
            self._heappush(pair, counts)

        for i in range(256, self.vocab_size):
            self.iter_BPE(i)
        
        for i, tok in enumerate(self.special_tokens):
            self.vocab[self.vocab_size + i] = tok.encode("utf-8")
        
        return self.vocab, self.merges
    


if __name__ == "__main__":

    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]

    bpe = BPETokenizer(vocab_size=10_000, input_path=input_path, special_tokens=special_tokens)
    vocab, merges = bpe.train_BPE()
 
    print(len(vocab))
    print(merges[:32])

