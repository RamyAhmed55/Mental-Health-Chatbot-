import torch
import torch.nn as nn
from transformers import DistilBertModel


class EmotionClassifier(nn.Module):

    def __init__(self, num_classes=6, class_weights=None, threshold=0.9):
        super(EmotionClassifier, self).__init__()
        self.bert    = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(self.bert.config.hidden_size, num_classes)

        if class_weights is not None:

            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.threshold = threshold
        self.label_map = {
            0: "sadness", 1: "joy",  2: "love",
            3: "anger",   4: "fear", 5: "surprise"}

# =====================================================================================================


    def forward(self, input_ids, attention_mask, labels=None):

        outputs      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled       = outputs.last_hidden_state[:, 0]
        pooled       = self.dropout(pooled)
        logits       = self.fc(pooled)
        loss         = self.loss_fn(logits, labels) if labels is not None else None

        return logits, loss
    
# =====================================================================================================

    def predict(self, input_ids, attention_mask):

        self.eval()
        with torch.no_grad():

            logits, _ = self.forward(input_ids, attention_mask)
            probs     = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            results = []

            for prob, pred in zip(max_probs, preds):

                if prob < self.threshold:
                    results.append("normal")
                    
                else:
                    results.append(self.label_map[int(pred)])
            return results