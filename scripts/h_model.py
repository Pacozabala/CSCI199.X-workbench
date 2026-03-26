import torch
import torch.nn as nn
from transformers import RobertaModel

# =========================
# MODEL
# =========================
'''
Defines the Hierarchical Model, which inherits nn.Module
- forward() function: used in training, returns loss and logits.
''' 
class HierarchicalRoBERTa(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super().__init__()

        self.lambda_weight = lambda_weight
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        hidden = self.encoder.config.hidden_size

        self.foundation_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(5)]
        )

        self.polarity_heads = nn.ModuleList(
            [nn.Linear(hidden, 3) for _ in range(5)]
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids, attention_mask,
                foundation_labels=None,
                polarity_labels=None):
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # get the 1st (CLS) token
        pooled = outputs.last_hidden_state[:,0]

        # flatten all found. logits into one array
        foundation_logits = torch.cat(
            [head(pooled) for head in self.foundation_heads],
            dim=1
        )


        polarity_logits = torch.stack(
            [head(pooled) for head in self.polarity_heads],
            dim=1
        ) # constructs the 3D array with size: [batch, 5, 3] -> batch entries, 5 triples per entry

        loss = None

        if foundation_labels is not None:

            foundation_loss = self.bce(
                foundation_logits,
                foundation_labels
            )

            polarity_loss = 0
            for f in range(5):
                mask = foundation_labels[:, f]
                ce_loss = self.ce(
                    polarity_logits[:, f, :],
                    polarity_labels[:, f]
                )
                
                masked_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
                polarity_loss += masked_loss
            
            # masked overall loss
            loss = foundation_loss + (self.lambda_weight* polarity_loss)

        return loss, foundation_logits, polarity_logits
    