from fastapi import FastAPI
import torch
import joblib
import re
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = FastAPI()

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("model/intent_classifier.pt"))
model.eval()

tokenizer = joblib.load("model/tokenizer.pkl")

# Define intent labels
intent_labels = {0: "informational", 1: "transactional"}

class QueryRequest(BaseModel):
    query: str

def extract_entity(query):
    """ Extracts potential entities from transactional queries """
    entity_patterns = {
        "leave": r"\b(leave|vacation|holiday)\b",
        "date": r"\b(\d{1,2} [A-Za-z]+|\bMonday\b|\bTuesday\b|\bnext week\b)\b",
        "reimbursement": r"\b(reimbursement|expense|travel claim)\b",
        "account": r"\b(bank|account|salary|update account)\b",
    }
    
    entities = {}
    for key, pattern in entity_patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities[key] = match.group(0)
    
    return entities if entities else None

@app.post("/classify/")
def classify_query(request: QueryRequest):
    inputs = tokenizer(request.query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs).logits
        prediction = torch.argmax(output, dim=1).item()
    
    predicted_intent = intent_labels[prediction]

    # If transactional, extract entities
    if predicted_intent == "transactional":
        entities = extract_entity(request.query)
        return {"intent": predicted_intent, "entities": entities}

    # Otherwise, return the query as is
    return {"response": request.query}
