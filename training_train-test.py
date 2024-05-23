import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
import json
import torch.nn as nn
import os

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Set environment variable to avoid OpenBLAS warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Set CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = 'augmented_dataset_small'
writer = SummaryWriter(f'runs/{dataset_name}')
# Load dataset (replace 'dataset_name' with your actual dataset)
dataset_tr = load_dataset('json', data_files=f'./data/{dataset_name}_train.jsonl')
dataset_te = load_dataset('json', data_files=f'./data/{dataset_name}_test.jsonl')

dataset_tr = dataset_tr.shuffle(seed=42)
dataset_te = dataset_te.shuffle(seed=42)

# Define the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_hidden_states=True)  # Adjust num_labels according to your dataset
model.to(device)
model.train

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['review'], truncation=True, padding='max_length', max_length=128)

def format_dataset(dataset):
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    return dataset['train']

train_dataset = format_dataset(dataset_tr)
test_dataset = format_dataset(dataset_te)

# Prepare DataLoader
batch_size = 64 #16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5) #lr=5e-5
num_epochs = 60
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Define the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    losses = []
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        inputs_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        labels=labels.long()
        attention = batch['attention_mask'].to(device)

        outputs = model(input_ids=inputs_ids, labels=labels, attention_mask = attention)
        loss = outputs.loss
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
        lr_scheduler.step()
    
    print(f"Epoch {epoch + 1} completed")
    losses = torch.tensor(losses)
    writer.add_scalar('Loss/validation', torch.mean(losses), epoch)

model.eval()
all_predictions = []
all_labels = []
features = []
num_classes = 2

for batch in eval_dataloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention)
        # Extract the hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]
        features.extend(hidden_states.cpu().numpy())

    predictions = outputs.logits.argmax(dim=-1)
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Convert features to numpy array
features = np.array(features)

# Reshape features to 2D (6, 128*768)
features = features.reshape(features.shape[0], -1)

# Check the number of samples
num_samples = features.shape[0]

# Set perplexity to a value less than the number of samples
perplexity_value = min(30, num_samples - 1)  # Example value, can be adjusted
# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, ticks=range(num_classes))
plt.title('t-SNE visualization of BERT features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

title = f'3_paraphrased.png'
plt.title(title)
# Generate a unique file name for the plot
# plot_path = os.path.join("./results", title)
# if not os.path.exists(plot_path):
#     os.makedirs(plot_path)

# Save the plot
plt.savefig(f'./results/tsne_{dataset_name}.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to free up memory

# Calculate metrics
accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
f1 = f1_score(all_labels, all_predictions, average=None)
conf_matrix = confusion_matrix(all_labels, all_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

