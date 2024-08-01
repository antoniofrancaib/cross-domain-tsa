# # Cross-Domain Sentiment Classification with Domain-Adaptive Neural Networks

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

from common_utils import clean_text

import pandas as pd
import numpy as np

import json

from torch.utils.data import DataLoader, TensorDataset

import string

from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import torch.optim as optim

# # Initialize Dataframes

# hyperparameters
yelp_sample_size = 10000

# ## IMDB Data ∼ Domain Source
# In[1]:

# In[2]:

imdb_df = pd.read_csv('IMDB Dataset.csv')
imdb_df.head()

# In[3]:

sentiment_counts_imdb = imdb_df['sentiment'].value_counts()
print(sentiment_counts_imdb)

# ## Yelp Data ∼ Target Source
# In[5]:

data_file = open("yelp_academic_dataset_review.json")
review_df = []
for line in data_file:
    review_df.append(json.loads(line))
yelp_df = pd.DataFrame(review_df)
data_file.close()

# In[6]:

# Filter out rows where 'stars' is 3
yelp_df = yelp_df[yelp_df['stars'] != 3.0].copy()

yelp_df['sentiment'] = yelp_df['stars'].apply(lambda x: 'positive' if x >= 4 else 'negative')
yelp_df = yelp_df.rename(columns={'text': 'review'})

yelp_df = yelp_df[['review', 'sentiment']]


# sampling an equal number of positive and negative reviews
yelp_positive_sample = yelp_df[yelp_df['sentiment'] == 'positive'].sample(n=yelp_sample_size // 2, random_state=42)
yelp_negative_sample = yelp_df[yelp_df['sentiment'] == 'negative'].sample(n=yelp_sample_size // 2, random_state=42)

yelp_balanced_sample = pd.concat([yelp_positive_sample, yelp_negative_sample]).sample(frac=1, random_state=42)

sentiment_counts = yelp_balanced_sample['sentiment'].value_counts()
print(sentiment_counts)

# ## Combine Data to create the final Training Dataset

imdb_df['domain'] = 0
yelp_balanced_sample['domain'] = 1

combined_df = pd.concat([imdb_df, yelp_balanced_sample])
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # 1. Initial Model Training with Source Domain (IMDB)

# ## Data Preprocessing for IMDB:

# ### Clean & Normalization: removing stopwords, special characters, stemming, and lemmatization.
combined_df['review'] = combined_df['review'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(list(combined_df['review']), padding=True, truncation=True, return_tensors="pt", max_length=512)


sentiment_mapping = {'positive': 1, 'negative': 0}

combined_df['sentiment'] = combined_df['sentiment'].map(sentiment_mapping)

sentiments = torch.tensor(combined_df['sentiment'].values, dtype=torch.long)
domains = torch.tensor(combined_df['domain'].values, dtype=torch.long)


full_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], sentiments, domains)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ## Build Model:

# Define the model architecture
class SentimentDomainModel(nn.Module):
    def __init__(self, bert_model_name, hidden_size, sentiment_classes, domain_classes):
        super(SentimentDomainModel, self).__init__()

        # Feature Extractor
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Sentiment Classifier
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, sentiment_classes)
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, domain_classes)
        )

        self.gradient_reversal_alpha = 1.0  # Define the negative constant for the gradient reversal layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT Feature Extraction
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs.pooler_output

        # Apply the gradient reversal layer with the chosen alpha
        reversed_features = gradient_reversal(pooled_output, self.gradient_reversal_alpha)

        # Sentiment classification
        sentiment_output = self.sentiment_classifier(pooled_output)

        # Domain classification
        domain_output = self.domain_classifier(reversed_features)

        return sentiment_output, domain_output


# Initialize the model
bert_model_name = 'bert-base-uncased'  # You can choose other BERT models as needed
hidden_size = 128  # Size of the hidden layer for both classifiers
sentiment_classes = 2  # Assuming binary classification for sentiment
domain_classes = 2  # Assuming binary classification for domain

model = SentimentDomainModel(bert_model_name, hidden_size, sentiment_classes, domain_classes)

# In[ ]:



# Assuming you have already defined the SentimentDomainModel as above

# Criterion for sentiment and domain classification
sentiment_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Number of epochs
num_epochs = 3


# Training and evaluation function
def train_model(model, train_loader, val_loader, num_epochs, sentiment_criterion, domain_criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_sentiment_loss = 0
        total_domain_loss = 0

        for batch in train_loader:
            # Unpack the batch
            input_ids, attention_mask, sentiments, domains = batch

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            sentiment_outputs, domain_outputs = model(input_ids, attention_mask, None)

            # Compute loss
            sentiment_loss = sentiment_criterion(sentiment_outputs, sentiments)
            domain_loss = domain_criterion(domain_outputs, domains)

            # Combine losses and backward pass
            total_loss = sentiment_loss + domain_loss
            total_loss.backward()

            # Update parameters
            optimizer.step()

            # Statistics
            total_sentiment_loss += sentiment_loss.item()
            total_domain_loss += domain_loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, sentiments, domains = batch

                # Forward pass
                sentiment_outputs, domain_outputs = model(input_ids, attention_mask, None)

                # Compute loss
                sentiment_loss = sentiment_criterion(sentiment_outputs, sentiments)
                domain_loss = domain_criterion(domain_outputs, domains)
                val_loss = sentiment_loss + domain_loss

                total_val_loss += val_loss.item()

                # Sentiment accuracy
                _, predicted = torch.max(sentiment_outputs.data, 1)
                correct += (predicted == sentiments).sum().item()

        val_accuracy = correct / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: Sentiment {total_sentiment_loss:.4f} Domain {total_domain_loss:.4f}, '
              f'Val Loss: {total_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), 'sentiment_domain_model.pth')


# Call the training function
train_model(model, train_loader, val_loader, num_epochs, sentiment_criterion, domain_criterion, optimizer)