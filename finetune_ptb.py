# finetune_ptb.py

import math
from pathlib import Path

import torch
import torch.nn as nn

from data_utils import seed_everything, PTBCorpus, batchify, get_batch
from models import TransformerModel, generate_square_subsequent_mask


def main() -> None:
    seed = 42
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Update this to your local PTB folder that contains ptb.train.txt/ptb.valid.txt/ptb.test.txt
    DATA_DIR_PTB = "Datasets/PTB"
    p = Path(DATA_DIR_PTB)
    if not p.exists():
        raise FileNotFoundError(f"DATA_DIR_PTB not found: {DATA_DIR_PTB}. Please extract PTB into this folder.")

    ptb_corpus = PTBCorpus(DATA_DIR_PTB, device=device)
    ptb_vocab_size = len(ptb_corpus.dictionary)
    print("PTB vocab size:", ptb_vocab_size)

    batch_size = 20
    eval_batch_size = 10
    bptt = 70

    ptb_train_data = batchify(ptb_corpus.train, batch_size)
    ptb_val_data = batchify(ptb_corpus.valid, eval_batch_size)
    ptb_test_data = batchify(ptb_corpus.test, eval_batch_size)

    # Load the pretrained WikiText-2 checkpoint
    WIKI_CKPT_PATH = "best_model.pt"
    checkpoint = torch.load(WIKI_CKPT_PATH, map_location=device)

    d_model = checkpoint["d_model"]
    nhead = checkpoint["nhead"]
    d_hid = checkpoint["d_hid"]
    nlayers = checkpoint["nlayers"]
    dropout = checkpoint["dropout"]
    wiki_state_dict = checkpoint["model_state_dict"]

    print(f"Loaded checkpoint from {WIKI_CKPT_PATH}")
    print(f"Architecture: {nlayers} layers, {nhead} heads, d_model={d_model}")

    ptb_model = TransformerModel(ptb_vocab_size, d_model, nhead, d_hid, nlayers, dropout).to(device)

    # Transfer matching layers only
    ptb_model_dict = ptb_model.state_dict()
    transferred_weights = {k: v for k, v in wiki_state_dict.items() if k in ptb_model_dict and v.size() == ptb_model_dict[k].size()}
    ptb_model_dict.update(transferred_weights)
    ptb_model.load_state_dict(ptb_model_dict)

    print(f"Methodology: Successfully transferred {len(transferred_weights)} layers.")

    # Freeze transferred layers
    for name, param in ptb_model.named_parameters():
        if name in transferred_weights:
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    ptb_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, ptb_model.parameters()), lr=0.0005, weight_decay=0.01)
    ptb_model.decoder.weight = ptb_model.encoder.weight
    ptb_scheduler = torch.optim.lr_scheduler.StepLR(ptb_optimizer, step_size=1, gamma=0.95)

    best_ptb_val_loss = float("inf")
    finetuned_model_path = "finetuned_ptb_best.pt"
    patience = 3
    trigger_times = 0

    for epoch in range(1, 11):
        ptb_model.train()
        total_train_loss = 0.0

        for _, i in enumerate(range(0, ptb_train_data.size(0) - 1, bptt)):
            data, targets = get_batch(ptb_train_data, i, bptt)
            src_mask = generate_square_subsequent_mask(data.size(0), device=torch.device(device))

            ptb_optimizer.zero_grad()
            output = ptb_model(data, src_mask)
            loss = criterion(output.view(-1, ptb_vocab_size), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ptb_model.parameters(), 0.5)
            ptb_optimizer.step()
            total_train_loss += loss.item()

        ptb_scheduler.step()

        ptb_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i in range(0, ptb_val_data.size(0) - 1, bptt):
                data, targets = get_batch(ptb_val_data, i, bptt)
                src_mask = generate_square_subsequent_mask(data.size(0), device=torch.device(device))
                output = ptb_model(data, src_mask)
                total_val_loss += len(data) * criterion(output.view(-1, ptb_vocab_size), targets).item()

        val_loss = total_val_loss / (len(ptb_val_data) - 1)
        print(f"| Epoch {epoch:3d} | Valid PPL {math.exp(val_loss):8.2f} | LR {ptb_scheduler.get_last_lr()[0]:.6f} |")

        if val_loss < best_ptb_val_loss:
            best_ptb_val_loss = val_loss
            torch.save(ptb_model.state_dict(), finetuned_model_path)
            print(f"Saved best finetuned model to {finetuned_model_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stopping Triggered!")
                break

    # Test
    ptb_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    ptb_model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for i in range(0, ptb_test_data.size(0) - 1, bptt):
            data, targets = get_batch(ptb_test_data, i, bptt)
            src_mask = generate_square_subsequent_mask(data.size(0), device=torch.device(device))
            output = ptb_model(data, src_mask)
            test_loss += len(data) * criterion(output.view(-1, ptb_vocab_size), targets).item()

    print(f"| Final PTB Test Perplexity: {math.exp(test_loss / (len(ptb_test_data) - 1)):8.2f} |")


if __name__ == "__main__":
    main()