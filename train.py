import torch
import pandas as pd
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss


# Sample labeled dataset
df = pd.read_csv("dataset.csv")
intent_labels = {"informational": 0, "transactional": 1}  # Convert labels to numbers
df["intent"] = df["intent"].map(intent_labels)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Custom dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item




# Prepare dataset


# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=3e-5, 
    weight_decay=0.01, 
    logging_steps=500
)

# ✅ Custom Trainer with Weighted Loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Assign higher weight to transactional (label=1) to balance the dataset
        loss_fct = CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]))  
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Load DistilBERT model for classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.classifier.weight.data.normal_(mean=0.0, std=0.02)
model.classifier.bias.data.zero_()

train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df["query"].tolist(), df["intent"].tolist(), test_size=0.2, random_state=42
)


# Create datasets
train_dataset = IntentDataset(train_texts, train_labels)
eval_dataset = IntentDataset(eval_texts, eval_labels)

# Train the model
trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()

# Save model and tokenizer
torch.save(model.state_dict(), "model/intent_classifier.pt")
joblib.dump(tokenizer, "model/tokenizer.pkl")

print("Model training complete. Saved as intent_classifier.pt")
