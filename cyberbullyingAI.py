import sys
!{sys.executable} -m pip install accelerate transformers[torch] -U

import torch 
from transformers import BertTokenizer, BertForSequenceClassification

print("--- ΕΛΕΓΧΟΣ ΣΥΣΤΗΜΑΤΟΣ ---") 
try:
    my_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    print("ΕΠΙΤΥΧΙΑ: ΤΟ BERT φορτώθηκε κανονικά")

    if torch.cuda.is.available():
       print("Η κάρτα γραφικών είναι έτοιμη για δουλειά")
    else:
    print("θα χρησιμοποιήσουμε τον επεξεργαστή.")
except Exception as σφάλμα:
    print(f"Κάτι πήγε λάθος: {σφάλμα}")

import pandas as pd 

try:
    df = pd.read_csv('cyberbullying_tweets.csv')
    print(" Το DATASET φορτώθηκε επιτυχώς. ")
    print(f"Συνολικά μηνύματα για ανάλυση: {len(df)}") 

    display(df.head())

    print("\nΠοσότητα ανά κατηγορία: ")
    print(df['cyberbullying_type'].value_counts())

except Exception as σφάλμα:
    print(f" Κάτι πήγε λάθος: {σφάλμα}") 

import re
import string 

def clean_text(text) 
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Ξεκινάει ο καθαρισμός των tweets... παρακαλώ περιμένετε")
df['cleaned_text'] = df['tweet_text'].apply(clean_text)

print("✅ Ο ΚΑΘΑΡΙΣΜΟΣ ΟΛΟΚΛΗΡΩΘΗΚΕ!") 

display(df[['tweet_text', 'cleaned_text']].head()) 

from transformers import BertTokenizer

my_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_example = df['cleaned_text'].iloc[0]
tokens = ny_tokenizer.toeknizer(text_example)
token_ids = my_tokenizer.convert_tokens_to_ids(tokens) 

print(f" Αρχικό κείμενο {text_example}") 
print(f"\n1. Tokens (Πωσ 'σπάει' τισ λέξεις): {tokens}"}
print(f"\n2. Tokens IDs (Πωσ βλέπει ο υπολογιστής): {token_ids}")

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Dataloader

df['label'] = df['cyberbullying_type'].astype('category').cat.codes
num_labels = len(df['label'].unique()) 

df_sample = df.sample(3000, randrom_state=42) 

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_sample['cleaned_text'].tolist(),
    df_sample['label'].tolist(),
    test_size=0.2
)

print(f"Δεδομένα έτοιμα! Εκπαίδευση με {len(train_texts)} tweets")

from transformers import Trainer, TrainerArguments, BertForSequenceClassification 

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

class CyberDataset(Dataset): 
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128) 
        self.labels = labels 
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) 
        return item 
    def __len__(self):
        return len(self.labels)

train_dataset = CyberDataset(train_texts, train_labels, my_tokenizer)
val_dataset = CyberDataset(val_texts, val_labels, my_tokenizer)

training_args = TrainingArguments(
  output_dir='./results',
  num_trains_epochs=3,
  per_device_train_batch_size=8,
  logging_steps=10,
  eval_strategy="epoch"
)

trainer = Trainer(
  model=model,
  args-training_args,
  train_dataset=train_dataset,
  eval_strategy=val_dataset
) 

print("Η εκπαίδευση ξεκινάει τώρα... Μπορεί να πάρει λίγη ώρα!")
trainer.train() 

def predict_cyberbullying(text):

    inputs = my_tokenizer(clean_text(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
  
    with torch,no_grad():
        outputs = model(**inputs) 
    prediction = torch.argmax(outputs.logits, dim=1).item()
    categoriew = df['cyberbullying_type'].astype('category').cat.categories

my_text =""
result = predict_cyberbullying(my_text)

print(f"Φράση: {my_text}")
print(f"Αποτέλεσμα ΑΙ: {result}")

import sys 
!{sys.executable} -m pip install seaborn matplotlib 

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib as plt 

predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1) 

cm = confusion_matrix(val_labels, preds)
categoriew - df['cyberbullying_type'].astype('category').cat.categories 

plt.figure(figsize=(10,8)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories, cmap='Blues')
plt.xlabel('Πρόβλεψη ΑΙ')
plt.ylabel('Πραγματικότητα')
plt.title('Πίνακασ Σύγχυσης (Confusion Matrix)')
plt.show()

print(classification_report(val_labels, preds, target_names=categories))  























