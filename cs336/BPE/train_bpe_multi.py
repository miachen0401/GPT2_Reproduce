from __future__ import annotations

import heapq
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Optional, Set, Tuple

import regex as re

Pair = Tuple[int, int]


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


_WORK_PAT: Optional[re.Pattern] = None


def _init_worker() -> None:
    """Initializer to compile regex once per worker process."""
    global _WORK_PAT
    _WORK_PAT = re.compile(PATTERN)


def _tokenize_docs_batch(docs: List[str]) -> CounterType[bytes]:
    """
    Worker function: tokenize a list of document strings into GPT-2 regex tokens,
    returning a Counter[bytes] for pre-tokens.
    """
    global _WORK_PAT
    assert _WORK_PAT is not None

    local: CounterType[bytes] = Counter()
    for doc in docs:
        for m in _WORK_PAT.finditer(doc):
            tok = m.group(0)
            if tok:
                local[tok.encode("utf-8")] += 1
    return local


def _invert_bytes(b: bytes) -> bytes:
    """
    Invert bytes for heap tie-break.

    We want: among same counts, choose lexicographically GREATER (b0, b1).
    Python heap pops smallest, so we push inverted bytes to reverse ordering.
    """
    return bytes(255 - x for x in b)


class BPETokenizer:
    def __init__(
        self,
        vocab_size: int = 10_000,
        input_path: str = "data/TinyStoriesV2-GPT4-train.txt",
        special_tokens: Optional[list[str]] = None,
        print_interval: int = 200,
    ) -> None:
        """
        Byte-level BPE trainer (efficient merges).

        Args:
            vocab_size: Max vocabulary size (excluding special tokens you add later if you do so outside).
            input_path: Path to the training corpus.
            special_tokens: Special tokens, e.g. ["<|endoftext|>"].
            print_interval: Print progress every N merge steps.
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: List[Tuple[bytes, bytes]] = []

        self.input_path = input_path
        self.special_tokens = special_tokens or []
        self.print_interval = print_interval

        if len(self.special_tokens) > 1:
            raise ValueError("This implementation expects at most 1 special token delimiter.")

        self._delimiter = self.special_tokens[0] if self.special_tokens else None

    def stream_freq_table_mp(
        self,
        chunk_size: int = 128 * 1024 * 1024,
        batch_docs: int = 5000,
        pretoken_workers: int = 4,
    ) -> CounterType[bytes]:
        """
        Stream the file, split by <|endoftext|> (document delimiter), and
        multiprocessing pretokenize the documents.

        Returns:
            Counter mapping pretoken bytes -> frequency.
        """
        file_size = os.path.getsize(self.input_path)
        total_chunks = max(1, math.ceil(file_size / chunk_size))

        if not self._delimiter:
            raise ValueError("TinyStories expects <|endoftext|> as delimiter; provide special_tokens.")

        delim = self._delimiter

        global_counter: CounterType[bytes] = Counter()

        carry = ""
        chunk_idx = 0
        docs_buffer: List[str] = []
        futures = []

        with ProcessPoolExecutor(
            max_workers=min(pretoken_workers, os.cpu_count() or 1),
            initializer=_init_worker,
        ) as ex, open(self.input_path, "r", encoding="utf-8", errors="ignore") as f:
            print(f"[BPE] Start streaming+pretokenize: {self.input_path}")

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                chunk_idx += 1
                if chunk_idx % 1 == 0:
                    print(f"[BPE] Read chunk {chunk_idx}/{total_chunks}")

                text = carry + chunk
                parts = text.split(delim)

                carry = parts.pop()  # incomplete tail (may contain partial delimiter)

                for doc in parts:
                    if not doc:
                        continue
                    docs_buffer.append(doc)
                    if len(docs_buffer) >= batch_docs:
                        futures.append(ex.submit(_tokenize_docs_batch, docs_buffer))
                        docs_buffer = []

            # flush remaining carry (as a doc if it has text besides delimiter)
            if carry.strip():
                docs_buffer.append(carry)

            if docs_buffer:
                futures.append(ex.submit(_tokenize_docs_batch, docs_buffer))

            # Collect progressively to avoid keeping many huge Counters in memory
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                local = fut.result()
                global_counter.update(local)
                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f"[BPE] Collected {done}/{total} pretoken batches")

        print(f"[BPE] Done pretokenization. Unique pretokens: {len(global_counter)}")
        return global_counter

    def _build_word_inventory(
        self,
        pretoken_counts: CounterType[bytes],
    ) -> Tuple[List[Tuple[int, ...]], List[int]]:
        """
        Convert Counter[bytes] to (words, freqs) using token-id sequences.

        Returns:
            words: list of token-id tuples
            freqs: list of frequencies aligned with words
        """
        words: List[Tuple[int, ...]] = []
        freqs: List[int] = []

        for b, c in pretoken_counts.items():
            if not b:
                continue
            words.append(tuple(b))  # bytes -> tuple[int]
            freqs.append(int(c))

        return words, freqs

    def _init_pair_stats(
        self,
        words: List[Tuple[int, ...]],
        freqs: List[int],
    ) -> Tuple[Dict[Pair, int], Dict[Pair, Set[int]]]:
        """
        Build initial pair_counts and pair_to_words index.

        pair_counts[p] = total frequency across corpus.
        pair_to_words[p] = set(word_id) containing that pair (unique membership).
        """
        pair_counts: Dict[Pair, int] = {}
        pair_to_words: Dict[Pair, Set[int]] = {}

        for wid, tokens in enumerate(words):
            n = len(tokens)
            if n < 2:
                continue

            seen_pairs: Set[Pair] = set()
            for i in range(n - 1):
                p = (tokens[i], tokens[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freqs[wid]
                seen_pairs.add(p)

            for p in seen_pairs:
                s = pair_to_words.get(p)
                if s is None:
                    pair_to_words[p] = {wid}
                else:
                    s.add(wid)

        return pair_counts, pair_to_words

    def _push_heap_entry(
        self,
        heap: List[Tuple[int, bytes, bytes, int, int]],
        pair: Pair,
        count: int,
    ) -> None:
        """Push a comparable heap entry with tie-break by lexicographically greater bytes pair."""
        if count <= 0:
            return

        b0 = self.vocab[pair[0]]
        b1 = self.vocab[pair[1]]

        heapq.heappush(
            heap,
            (-count, _invert_bytes(b0), _invert_bytes(b1), pair[0], pair[1]),
        )

    def _pop_best_pair(
        self,
        heap: List[Tuple[int, bytes, bytes, int, int]],
        pair_counts: Dict[Pair, int],
    ) -> Optional[Pair]:
        """
        Pop best pair using lazy validation.

        Heap item is:
            (neg_count, inv_b0, inv_b1, p0, p1)
        """
        while heap:
            neg_count, _inv_b0, _inv_b1, p0, p1 = heapq.heappop(heap)
            pair = (p0, p1)
            cur = pair_counts.get(pair, 0)
            if cur <= 0:
                continue
            if -neg_count != cur:
                continue
            return pair
        return None

    def _merge_word(
        self,
        tokens: Tuple[int, ...],
        pair: Pair,
        new_id: int,
    ) -> Tuple[int, ...]:
        """Merge all occurrences of pair in a token tuple, returning a new token tuple."""
        p0, p1 = pair
        n = len(tokens)
        if n < 2:
            return tokens

        out: List[int] = []
        i = 0
        while i < n - 1:
            if tokens[i] == p0 and tokens[i + 1] == p1:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        if i == n - 1:
            out.append(tokens[-1])
        return tuple(out)

    def _pairs_in_word(self, tokens: Tuple[int, ...]) -> Set[Pair]:
        """Return set of unique adjacent pairs in a token tuple."""
        n = len(tokens)
        if n < 2:
            return set()
        return {(tokens[i], tokens[i + 1]) for i in range(n - 1)}

    def train(
        self,
        chunk_size: int = 256 * 1024 * 1024,
        batch_docs: int = 5000,
        pretoken_workers: int = 4,
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Train BPE up to self.vocab_size using:
        - MP pretokenization (doc-split by <|endoftext|>)
        - Incremental pair counting + heap (fast merges)

        Returns:
            vocab, merges
        """
        pretoken_counts = self.stream_freq_table_mp(
            chunk_size=chunk_size,
            batch_docs=batch_docs,
            pretoken_workers=pretoken_workers,
        )

        words, freqs = self._build_word_inventory(pretoken_counts)
        pair_counts, pair_to_words = self._init_pair_stats(words, freqs)

        heap: List[Tuple[int, bytes, bytes, int, int]] = []

        for p, c in pair_counts.items():
            self._push_heap_entry(heap, p, c)

        target_merges = max(0, self.vocab_size - 256)
        print(f"[BPE] Start merges: {target_merges} merges (vocab_size={self.vocab_size})")

        for step in range(target_merges):
            vocab_id = 256 + step
            pair = self._pop_best_pair(heap, pair_counts)
            if pair is None:
                print("[BPE] No more pairs to merge.")
                break

            # Record merge (bytes)
            self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))
            self.vocab[vocab_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            affected = pair_to_words.get(pair, set()).copy()
            if not affected:
                pair_counts[pair] = 0
                continue

            # We'll rebuild membership for affected words: remove old pairs, add new pairs
            for wid in affected:
                old_tokens = words[wid]
                old_pairs = self._pairs_in_word(old_tokens)

                new_tokens = self._merge_word(old_tokens, pair, vocab_id)
                words[wid] = new_tokens
                new_pairs = self._pairs_in_word(new_tokens)

                # Update pair_to_words memberships
                for p in old_pairs:
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(wid)
                        if not s:
                            pair_to_words.pop(p, None)

                for p in new_pairs:
                    s = pair_to_words.get(p)
                    if s is None:
                        pair_to_words[p] = {wid}
                    else:
                        s.add(wid)

                # Update pair_counts: subtract old, add new (weighted by word frequency)
                wfreq = freqs[wid]
                for p in old_pairs:
                    pair_counts[p] = pair_counts.get(p, 0) - wfreq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)

                for p in new_pairs:
                    pair_counts[p] = pair_counts.get(p, 0) + wfreq

                    # push updated count to heap (lazy)
                    self._push_heap_entry(heap, p, pair_counts[p])

            # This merged pair should be gone as a candidate
            pair_counts.pop(pair, None)
            pair_to_words.pop(pair, None)

            if (step + 1) % self.print_interval == 0 or (step + 1) == target_merges:
                print(f"[BPE] merge {step+1}/{target_merges}")

        return self.vocab, self.merges


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]

    bpe = BPETokenizer(
        vocab_size=10_000,
        input_path=input_path,
        special_tokens=special_tokens,
        print_interval=500,
    )

    vocab, merges = bpe.train(
        chunk_size=256 * 1024 * 1024,
        batch_docs=5000,
        pretoken_workers=12,
    )

    # Example: longest token
    longest_id = max(vocab.keys(), key=lambda i: len(vocab[i]))
    print(f"[BPE] longest token id={longest_id}, bytes_len={len(vocab[longest_id])}")
