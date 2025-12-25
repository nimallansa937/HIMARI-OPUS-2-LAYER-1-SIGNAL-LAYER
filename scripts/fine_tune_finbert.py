"""
FinBERT Fine-Tuning Pipeline for Crypto Sentiment

Enhancement 4 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md

Usage:
    python scripts/fine_tune_finbert.py --symbol BTCUSDT --num_examples 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

# Fine-tuning configuration
@dataclass
class FineTuningConfig:
    """Configuration for FinBERT fine-tuning."""
    base_model: str = "ProsusAI/finbert"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_seq_length: int = 128
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    output_path: str = "./models/finbert-crypto-finetuned"
    

def load_labeled_data(data_path: str) -> List[Dict]:
    """
    Load labeled crypto headlines for fine-tuning.
    
    Expected format (JSONL):
        {"text": "Bitcoin surges to ATH", "label": 1}
        {"text": "Crypto market crashes", "label": -1}
        {"text": "Bitcoin trades sideways", "label": 0}
    
    Labels:
        1 = bullish
        0 = neutral
        -1 = bearish
    """
    data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
        logger.info(f"Loaded {len(data)} labeled examples from {data_path}")
    except FileNotFoundError:
        logger.warning(f"Data file not found: {data_path}")
    
    return data


def split_data(
    data: List[Dict],
    config: FineTuningConfig
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets."""
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * config.train_split)
    val_end = int(n * (config.train_split + config.val_split))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def validate_label_distribution(data: List[Dict]) -> Dict[str, float]:
    """Check class balance in labeled data."""
    labels = [d['label'] for d in data]
    
    total = len(labels)
    bullish = sum(1 for l in labels if l == 1) / total
    bearish = sum(1 for l in labels if l == -1) / total
    neutral = sum(1 for l in labels if l == 0) / total
    
    dist = {
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral
    }
    
    logger.info(f"Label distribution: bullish={bullish:.1%}, bearish={bearish:.1%}, neutral={neutral:.1%}")
    
    # Warn if imbalanced
    if bullish < 0.25 or bearish < 0.25:
        logger.warning("Label distribution is imbalanced - consider resampling")
    
    return dist


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    config: FineTuningConfig
) -> Optional[str]:
    """
    Fine-tune FinBERT on crypto sentiment data.
    
    Requires transformers and torch installed.
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback
        )
        from datasets import Dataset
        import torch
    except ImportError as e:
        logger.error(f"Fine-tuning requires transformers and datasets: {e}")
        return None
    
    logger.info(f"Loading base model: {config.base_model}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=3,  # positive, negative, neutral
        ignore_mismatched_sizes=True
    )
    
    # Prepare datasets
    def prepare_dataset(data: List[Dict]) -> Dataset:
        texts = [d['text'] for d in data]
        # Convert labels: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
        labels = [d['label'] + 1 for d in data]
        
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.max_seq_length
        )
        
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
    
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        logging_dir=f"{config.output_path}/logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save model
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    
    logger.info(f"Model saved to: {config.output_path}")
    
    return config.output_path


def evaluate_model(
    test_data: List[Dict],
    model_path: str
) -> Dict[str, float]:
    """Evaluate fine-tuned model on test set."""
    try:
        from transformers import pipeline
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    except ImportError as e:
        logger.error(f"Evaluation requires transformers and sklearn: {e}")
        return {}
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Load fine-tuned model
    classifier = pipeline("sentiment-analysis", model=model_path)
    
    # Predict
    predictions = []
    for item in test_data:
        result = classifier(item['text'])[0]
        
        # Convert back to -1/0/1
        label = result['label'].lower()
        if 'positive' in label:
            pred = 1
        elif 'negative' in label:
            pred = -1
        else:
            pred = 0
        
        predictions.append(pred)
    
    # Ground truth
    labels = [d['label'] for d in test_data]
    
    # Metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'test_size': len(test_data)
    }
    
    logger.info(f"Accuracy: {accuracy:.2%}, F1 (macro): {f1_macro:.3f}")
    
    return metrics


def create_sample_data(output_path: str, num_samples: int = 50):
    """Create sample labeled data for testing the pipeline."""
    
    bullish_phrases = [
        "Bitcoin surges past $100K as institutional adoption grows",
        "ETH breaking out, bulls taking control",
        "Crypto market recovery accelerates, BTC leading the charge",
        "Massive accumulation by whales detected on-chain",
        "SEC approves Bitcoin ETF, prices soar",
        "Bull run confirmed as BTC breaks key resistance",
        "Diamond hands paying off as prices moon",
        "WAGMI - market sentiment extremely bullish",
    ]
    
    bearish_phrases = [
        "Bitcoin crashes below support, panic selling ensues",
        "Crypto market in freefall, billions liquidated",
        "Exchange hack causes massive selloff",
        "Regulatory crackdown sends prices tumbling",
        "Bear market deepens as BTC tests new lows",
        "Getting rekt on this dump, total capitulation",
        "FUD spreading as major exchange suspends withdrawals",
        "NGMI - market sentiment extremely bearish",
    ]
    
    neutral_phrases = [
        "Bitcoin consolidates around $50K level",
        "Market trades sideways with low volume",
        "Analysts divided on near-term direction",
        "BTC fluctuates within established range",
        "Mixed signals as market awaits macro data",
        "Traders cautious ahead of Fed announcement",
        "Volume decreases as market takes a breather",
        "No clear trend as prices remain stable",
    ]
    
    samples = []
    
    # Generate balanced samples
    for i in range(num_samples // 3):
        samples.append({'text': random.choice(bullish_phrases), 'label': 1})
        samples.append({'text': random.choice(bearish_phrases), 'label': -1})
        samples.append({'text': random.choice(neutral_phrases), 'label': 0})
    
    random.shuffle(samples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Created {len(samples)} sample examples at {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Fine-tune FinBERT for crypto sentiment')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--data_path', type=str, default='./data/fine_tuning/crypto_headlines.jsonl')
    parser.add_argument('--output_path', type=str, default='./models/finbert-crypto-finetuned')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--create_sample', action='store_true', help='Create sample data')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("FINBERT FINE-TUNING PIPELINE")
    print("=" * 70)
    
    config = FineTuningConfig(
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    # Create sample data if requested or if file doesn't exist
    if args.create_sample or not Path(args.data_path).exists():
        Path(args.data_path).parent.mkdir(parents=True, exist_ok=True)
        create_sample_data(args.data_path, num_samples=60)
    
    # Load data
    data = load_labeled_data(args.data_path)
    if not data:
        print("No data to train on. Use --create_sample to generate sample data.")
        return 1
    
    # Validate distribution
    validate_label_distribution(data)
    
    # Split
    train_data, val_data, test_data = split_data(data, config)
    
    if len(train_data) < 20:
        print(f"Insufficient training data ({len(train_data)} examples). Need at least 20.")
        return 1
    
    # Train
    model_path = train_model(train_data, val_data, config)
    
    if model_path:
        # Evaluate
        metrics = evaluate_model(test_data, model_path)
        
        print("\n" + "=" * 70)
        print("FINE-TUNING COMPLETE")
        print("=" * 70)
        print(f"Model saved: {model_path}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"F1 Score: {metrics.get('f1_macro', 0):.3f}")
        
        return 0
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
