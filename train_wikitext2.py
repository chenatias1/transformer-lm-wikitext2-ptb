# train_wikitext2.py

import time
import math
from pathlib import Path

import torch
import torch.nn as nn

from data_utils import seed_everything, Corpus, batchify, get_batch
from models import TransformerModel, generate_square_subsequent_mask


def evaluate(model: nn.Module, criterion: nn.Module, eval_data: torch.Tensor, bptt: int, ntokens: int, device: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            src_mask = generate_square_subsequent_mask(data.size(0), device=torch.device(device))
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    bptt: int,
    ntokens: int,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(bptt, device=torch.device(device))

    for _, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        if data.size(0) != bptt:
            src_mask = generate_square_subsequent_mask(data.size(0), device=torch.device(device))

        optimizer.zero_grad()
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (len(train_data) // bptt)


def main() -> None:
    seed = 42
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Update this path to your local extracted folder that contains wiki.train.tokens/wiki.valid.tokens/wiki.test.tokens
    DATA_DIR = "Datasets/wikitext-2"
    p = Path(DATA_DIR)
    if not p.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}. Please extract WikiText-2 into this folder.")

    corpus = Corpus(DATA_DIR, device=device)
    vocab_size = len(corpus.dictionary)
    print("Vocabulary size:", vocab_size)

    batch_size = 20
    eval_batch_size = 10
    bptt = 70

    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    d_model = 256
    nhead = 8
    d_hid = 512
    nlayers = 4
    dropout = 0.1

    model = TransformerModel(vocab_size, d_model, nhead, d_hid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_val_loss = float("inf")
    best_model_path = "best_model.pt"

    epochs = 20
    patience = 3
    trigger_times = 0

    train_ppls, val_ppls, lr_decay_epochs = [], [], []

    print("-" * 30)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(model, criterion, optimizer, train_data, bptt, vocab_size, device)
        val_loss = evaluate(model, criterion, val_data, bptt, vocab_size, device)

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if optimizer.param_groups[0]["lr"] < old_lr:
            lr_decay_epochs.append(epoch)

        print(
            f"| End of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | "
            f"valid ppl {val_ppl:8.2f} | lr {old_lr:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "d_model": d_model,
                "nhead": nhead,
                "d_hid": d_hid,
                "nlayers": nlayers,
                "dropout": dropout,
            }
            torch.save(checkpoint, best_model_path)
            print(f"> Found better model. Saving to {best_model_path} with metadata.")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"> No improvement. Early Stopping counter: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"!!! Early stopping at epoch {epoch}. Avoiding redundant runtime.")
                break

        print("-" * 30)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Best model weights loaded successfully from checkpoint.")

    test_loss = evaluate(model, criterion, test_data, bptt, vocab_size, device)
    test_ppl = math.exp(test_loss)
    print("=" * 30)
    print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f} |")
    print("=" * 30)


if __name__ == "__main__":
    main()