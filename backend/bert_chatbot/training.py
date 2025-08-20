"""
Training pipeline for BERT-based chatbot
Fine-tunes on API documentation and developer queries
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class APIDocumentationDataset(Dataset):
    """Dataset for API documentation and queries"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data from various sources"""
        data = []
        
        # Load from API specifications
        api_specs = Path(data_path) / "api_specs"
        for spec_file in api_specs.glob("*.json"):
            with open(spec_file) as f:
                spec = json.load(f)
                data.extend(self._extract_training_samples(spec))
        
        # Load from query logs (if available)
        query_logs = Path(data_path) / "query_logs.jsonl"
        if query_logs.exists():
            with open(query_logs) as f:
                for line in f:
                    data.append(json.loads(line))
        
        # Load from cookbook recipes
        recipes = Path(data_path) / "cookbook_recipes.json"
        if recipes.exists():
            with open(recipes) as f:
                recipe_data = json.load(f)
                data.extend(self._extract_recipe_samples(recipe_data))
        
        return data
    
    def _extract_training_samples(self, spec: Dict) -> List[Dict]:
        """Extract training samples from API specification"""
        samples = []
        
        for path, methods in spec.get('paths', {}).items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    # Create positive samples
                    sample = {
                        'query': f"How do I use {method.upper()} {path}?",
                        'api_endpoint': f"{method.upper()} {path}",
                        'intent': 'get_example',
                        'relevant': True
                    }
                    samples.append(sample)
                    
                    # Create samples for parameters
                    for param in details.get('parameters', []):
                        param_sample = {
                            'query': f"What is the {param['name']} parameter for {path}?",
                            'api_endpoint': f"{method.upper()} {path}",
                            'intent': 'understand_params',
                            'relevant': True
                        }
                        samples.append(param_sample)
        
        return samples
    
    def _extract_recipe_samples(self, recipes: Dict) -> List[Dict]:
        """Extract training samples from cookbook recipes"""
        samples = []
        
        for recipe_id, recipe in recipes.items():
            # Create samples from recipe questions
            for question in recipe.get('common_questions', []):
                sample = {
                    'query': question,
                    'api_endpoint': recipe.get('endpoint'),
                    'intent': recipe.get('intent_type', 'get_example'),
                    'relevant': True
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize query
        query_encoding = self.tokenizer(
            sample['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        intent_map = {
            'get_example': 0,
            'find_api': 1,
            'debug_error': 2,
            'understand_params': 3,
            'best_practices': 4,
            'authentication': 5
        }
        intent_label = intent_map.get(sample.get('intent', 'get_example'), 0)
        
        return {
            'input_ids': query_encoding['input_ids'].squeeze(),
            'attention_mask': query_encoding['attention_mask'].squeeze(),
            'intent_label': torch.tensor(intent_label),
            'relevance_label': torch.tensor(1.0 if sample.get('relevant', True) else 0.0)
        }

class BERTTrainer:
    """Trainer for BERT-based chatbot models"""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 10,
        device: str = None
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def train_intent_classifier(self, train_data: DataLoader, val_data: DataLoader):
        """Train the intent classification model"""
        # Build classifier head
        classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # 6 intent classes
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(
            list(self.base_model.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate
        )
        
        total_steps = len(train_data) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_accuracy = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.base_model.train()
            classifier.train()
            
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['intent_label'].to(self.device)
                
                # Forward pass
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
                logits = classifier(pooled_output)
                
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation phase
            val_accuracy = self._validate(val_data, classifier, criterion)
            
            # Print statistics
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss/len(train_data):.4f}")
            print(f"  Train Accuracy: {100*train_correct/train_total:.2f}%")
            print(f"  Val Accuracy: {100*val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model(classifier, 'intent_classifier')
    
    def _validate(self, val_data: DataLoader, classifier: nn.Module, criterion: nn.Module) -> float:
        """Validate the model"""
        self.base_model.eval()
        classifier.eval()
        
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_data:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['intent_label'].to(self.device)
                
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = classifier(pooled_output)
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        return val_correct / val_total
    
    def _save_model(self, model: nn.Module, name: str):
        """Save trained model"""
        save_path = Path("models") / f"{name}.pt"
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'base_model_name': self.model_name
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")

def create_training_data():
    """Create synthetic training data for the chatbot"""
    
    training_samples = []
    
    # Common query patterns
    query_templates = [
        ("How do I {action} a {resource}?", "get_example"),
        ("Show me an example of {endpoint}", "get_example"),
        ("What's the {language} code for {endpoint}?", "get_example"),
        ("I'm getting {error_code} error when calling {endpoint}", "debug_error"),
        ("Why is my {endpoint} request failing?", "debug_error"),
        ("What parameters does {endpoint} accept?", "understand_params"),
        ("What's the difference between {endpoint1} and {endpoint2}?", "compare_apis"),
        ("Best practices for {endpoint}", "best_practices"),
        ("How do I authenticate to {api}?", "authentication"),
        ("Rate limiting for {endpoint}", "rate_limiting")
    ]
    
    # API endpoints from BAnCS
    endpoints = [
        "GET /account/balance",
        "POST /customer/create",
        "GET /loan/details",
        "POST /transaction/transfer",
        "GET /account/transactions"
    ]
    
    # Generate samples
    for template, intent in query_templates:
        for endpoint in endpoints:
            # Create variations
            sample = {
                'query': template.format(
                    action='retrieve',
                    resource='account balance',
                    endpoint=endpoint,
                    language='Python',
                    error_code='401',
                    endpoint1=endpoint,
                    endpoint2=endpoints[0],
                    api='BAnCS'
                ),
                'intent': intent,
                'endpoint': endpoint
            }
            training_samples.append(sample)
    
    # Save training data
    output_path = Path("training_data/samples.jsonl")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + '\n')
    
    return training_samples

if __name__ == "__main__":
    # Create training data
    create_training_data()
    
    # Initialize trainer
    trainer = BERTTrainer()
    
    # Load dataset
    dataset = APIDocumentationDataset("training_data", trainer.tokenizer)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=trainer.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainer.batch_size)
    
    # Train model
    trainer.train_intent_classifier(train_loader, val_loader)
