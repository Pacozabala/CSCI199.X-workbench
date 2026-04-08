import time
from tqdm import tqdm
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import f1_score

# =========================
# TRAINING FUNCTION
# =========================
'''
Trains the model for 1 epoch.
'''
def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    start_time = time.time()

    loader_progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1}",
        leave=False
    )

    scaler = GradScaler()

    for batch in loader_progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        foundation_labels = batch["foundation_labels"].to(device)
        polarity_labels = batch["polarity_labels"].to(device)

        # gets rid of gradient from previous pass
        optimizer.zero_grad()

        # mixed precision forward() call
        with autocast():
            loss, _, _ = model(
                input_ids,
                attention_mask,
                foundation_labels,
                polarity_labels
            )
        

        # backpropagate the loss using scaler and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # update scaler
        scaler.update()

        total_loss += loss.item()

        loader_progress.set_postfix(loss=loss.item())

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader)

    return avg_loss, epoch_time

# =========================
# EVALUATION
# =========================
'''
Evaluates the model.
Returns foundation f1, polarity f1 per foundation, and mean polarity f1
'''
def evaluate(model, loader, device):
    model.eval()

    all_found_preds = []
    all_found_true = []

    all_pol_preds = [[] for _ in range(5)]
    all_pol_true = [[] for _ in range(5)]

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            foundation_labels = batch["foundation_labels"].to(device)
            polarity_labels = batch["polarity_labels"].to(device)

            _, foundation_logits, polarity_logits = model(
                input_ids, attention_mask
            )

            # foundation predictions
            found_preds = (torch.sigmoid(foundation_logits) > 0.5)

            all_found_preds.append(found_preds.cpu())
            all_found_true.append(foundation_labels.cpu())

            # polarity predictions (masked)
            pol_preds = torch.argmax(polarity_logits, dim=2)

            for f in range(5):
                mask_f = found_preds[:, f] == 1 # predicted foundations mask

                if mask_f.sum() > 0:
                    preds_f = pol_preds[mask_f, f]
                    true_f = polarity_labels[mask_f, f]

                    all_pol_preds[f].append(preds_f.cpu())
                    all_pol_true[f].append(true_f.cpu())
        
    foundation_f1 = f1_score(
        torch.cat(all_found_true).numpy().flatten(),
        torch.cat(all_found_preds).numpy().flatten(),
        average="macro"
    )

    polarity_f1_scores = []
    for f in range(5):
        if len(all_pol_true[f]) > 0:
            y_true = torch.cat(all_pol_true[f]).numpy()
            y_pred = torch.cat(all_pol_preds[f]).numpy()

            f1 = f1_score(y_true, y_pred, average="macro")
            polarity_f1_scores.append(f1)
        else:
            polarity_f1_scores.append(0.0)
    
    mean_polarity_f1 = sum(polarity_f1_scores) / 5

    return foundation_f1, polarity_f1_scores, mean_polarity_f1