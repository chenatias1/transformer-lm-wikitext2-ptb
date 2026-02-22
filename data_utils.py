# data_utils.py

import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(val: int) -> None:
    random.seed(val)
    os.environ["PYTHONHASHSEED"] = str(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)
    torch.backends.cudnn.deterministic = True
    print("Manual seed changed successfully.")


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus:
    """
    WikiText-2: load raw files, replace newlines with <eos>, tokenize to ids.
    """

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.dictionary = Dictionary()
        self.device = device
        data_dir = Path(data_dir)

        self.train = self._tokenize(data_dir / "wiki.train.tokens")
        self.valid = self._tokenize(data_dir / "wiki.valid.tokens")
        self.test = self._tokenize(data_dir / "wiki.test.tokens")

    def _tokenize(self, path: Path) -> torch.Tensor:
        text = path.read_text(encoding="utf-8")
        text = text.replace("\n", " <eos> ")
        words = text.split()

        ids = torch.empty(len(words), dtype=torch.long)
        for i, w in enumerate(words):
            ids[i] = self.dictionary.add_word(w)

        return ids.to(self.device)


class PTBCorpus:
    """
    Penn Treebank: same logic.
    """

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.dictionary = Dictionary()
        self.device = device
        data_dir = Path(data_dir)

        self.train = self._tokenize(data_dir / "ptb.train.txt")
        self.valid = self._tokenize(data_dir / "ptb.valid.txt")
        self.test = self._tokenize(data_dir / "ptb.test.txt")

    def _tokenize(self, path: Path) -> torch.Tensor:
        text = path.read_text(encoding="utf-8")
        text = text.replace("\n", " <eos> ")
        words = text.split()

        ids = torch.empty(len(words), dtype=torch.long)
        for i, w in enumerate(words):
            ids[i] = self.dictionary.add_word(w)

        return ids.to(self.device)


def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_batches = data.size(0) // batch_size
    data = data[: n_batches * batch_size]
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source: torch.Tensor, i: int, bptt: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target