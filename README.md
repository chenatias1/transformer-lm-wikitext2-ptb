# Transformers (from scratch + transfer learning)

This repository contains a Transformer language model trained from scratch on WikiText-2, and then fine-tuned on Penn Treebank (PTB) using transfer learning.

## Repository structure

- `models.py`
  - `PositionalEncoding`
  - `TransformerModel` (TransformerEncoder + causal mask for decoder-only behavior)
  - `generate_square_subsequent_mask`

- `data_utils.py`
  - Reproducibility: `seed_everything`
  - Tokenization and vocab: `Dictionary`, `Corpus` (WikiText-2), `PTBCorpus` (PTB)
  - Batching helpers: `batchify`, `get_batch`

- `train_wikitext2.py`
  - Train and evaluate on WikiText-2
  - Saves best checkpoint as `best_model.pt` (includes architecture metadata)

- `finetune_ptb.py`
  - Loads `best_model.pt`
  - Transfers only layers with matching shapes
  - Freezes transferred layers and trains new embedding/decoder for PTB
  - Saves best PTB model as `finetuned_ptb_best.pt`

## Requirements

Python 3.10+ recommended.

## Technologies
- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- AdamW optimizer
- StepLR scheduler
- Transformer architecture (decoder-only, causal masking)
- Transfer Learning
- Weight Tying
- Early Stopping
