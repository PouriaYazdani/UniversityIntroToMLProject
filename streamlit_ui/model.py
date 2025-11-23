import torch
from transformers import DistilBertModel, AutoConfig

class DistilBERTClass(torch.nn.Module):
    def __init__(self, num_labels=10, hidden_dropout=0.1, attention_dropout=0.1):
        super(DistilBERTClass, self).__init__()
        config = AutoConfig.from_pretrained(
            "distilbert-base-uncased",
            dropout=hidden_dropout,
            attention_dropout=attention_dropout
        )
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # (batch_size, seq_len, 768)
        pooler = hidden_state[:, 0]  # Extract CLS token
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        return self.classifier(pooler)
