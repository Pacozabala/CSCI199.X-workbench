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

    scaler = GradScaler(enabled=(device.type=="cuda"))

    for batch in loader_progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        foundation_labels = batch["foundation_labels"].to(device)
        polarity_labels = batch["polarity_labels"].to(device)

        # gets rid of gradient from previous pass
        optimizer.zero_grad()

        # mixed precision forward() call
        with autocast(device_type=device.type, enabled=(device.type=="cuda")):
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

            # model forward pass
            _, foundation_logits, polarity_logits = model(
                input_ids, attention_mask
            )

            # convert foundation logits to predictions 
            # sigmoid probs -> binary values
            found_preds = (torch.sigmoid(foundation_logits) > 0.5)

            all_found_preds.append(found_preds.cpu())
            all_found_true.append(foundation_labels.cpu())

            # convert polarity logits to predictions
            # choose the highest logit per foundation
            pol_preds = torch.argmax(polarity_logits, dim=2)

            # loop through foundations
            for f in range(5):
                mask_f = found_preds[:, f] == 1 # rows where foundation is predicted

                if mask_f.sum() > 0:
                    preds_f = pol_preds[mask_f, f]
                    true_f = polarity_labels[mask_f, f]

                    all_pol_preds[f].append(preds_f.cpu())
                    all_pol_true[f].append(true_f.cpu())
    
    found_y_true = torch.cat(all_found_true).numpy().flatten()
    found_y_pred = torch.cat(all_found_preds).numpy().flatten()

    found_macro_f1 = f1_score(found_y_true, found_y_pred, average="macro")
    found_micro_f1 = f1_score(found_y_true, found_y_pred, average="micro")

    pol_macro_scores = []
    pol_micro_scores = []
    for f in range(5):
        if len(all_pol_true[f]) > 0:
            y_true = torch.cat(all_pol_true[f]).numpy()
            y_pred = torch.cat(all_pol_preds[f]).numpy()

            macro_f1 = f1_score(y_true, y_pred, average="macro")
            micro_f1 = f1_score(y_true, y_pred, average="micro")
            
            pol_macro_scores.append(macro_f1)
            pol_micro_scores.append(micro_f1)
        else:
            pol_macro_scores.append(0.0)
            pol_micro_scores.append(0.0)
    
    mean_pol_macro = sum(pol_macro_scores) / 5
    mean_pol_micro = sum(pol_micro_scores) / 5

    return (
        found_macro_f1,
        found_micro_f1,
        pol_macro_scores,
        pol_micro_scores,
        mean_pol_macro,
        mean_pol_micro
    )