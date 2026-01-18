"""
Phase 1 Training: Next-action prediction (go/no-go validation).

This script trains the state model to predict the next action class
from the current state, validating that the recurrent state captures
useful information.
"""

import os
import argparse
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data.dataset import NextActionDataset, collate_episodes
from data.action_taxonomy import get_num_actions, get_action_name
from models.full_model import RecurrentStateModel


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_baseline_accuracy(dataset: NextActionDataset) -> float:
    """Compute accuracy of always predicting most common action."""
    from collections import Counter
    action_counts = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        action_counts[sample['target_action']] += 1

    most_common = action_counts.most_common(1)[0][1]
    return most_common / len(dataset)


def train_epoch(
    model: RecurrentStateModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_acc = 0
    total_baseline_acc = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        # Forward pass
        output = model(batch['events'])
        logits = output['logits']
        baseline_logits = output['baseline_logits']

        targets = batch['target_actions'].to(device)

        # Compute losses
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Metrics
        acc = compute_accuracy(logits, targets)
        baseline_acc = compute_accuracy(baseline_logits, targets)

        total_loss += loss.item()
        total_acc += acc
        total_baseline_acc += baseline_acc
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{acc:.2%}',
            'baseline': f'{baseline_acc:.2%}'
        })

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'baseline_accuracy': total_baseline_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: RecurrentStateModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    total_acc = 0
    total_baseline_acc = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        output = model(batch['events'])
        logits = output['logits']
        baseline_logits = output['baseline_logits']

        targets = batch['target_actions'].to(device)

        loss = criterion(logits, targets)
        acc = compute_accuracy(logits, targets)
        baseline_acc = compute_accuracy(baseline_logits, targets)

        total_loss += loss.item()
        total_acc += acc
        total_baseline_acc += baseline_acc
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'baseline_accuracy': total_baseline_acc / num_batches,
    }


def train(
    config_path: str,
    data_path: str,
    output_dir: str,
    use_wandb: bool = True,
):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'ssm-agent'),
            config=config,
            name=f"phase1_next_action",
        )

    # Load data
    env = config.get('env', 'webshop')
    num_actions = get_num_actions(env)

    full_dataset = NextActionDataset(
        episodes_path=data_path,
        env=env,
        max_episodes=config.get('max_episodes', None),
    )

    # Train/val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_episodes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_episodes,
    )

    # Compute baselines
    majority_baseline = compute_baseline_accuracy(full_dataset)
    print(f"Majority class baseline: {majority_baseline:.2%}")

    # Initialize model
    model = RecurrentStateModel(
        text_model=config['encoder']['text_model'],
        d_event=config['encoder']['d_event'],
        d_state=config['state_updater']['d_state'],
        num_gru_layers=config['state_updater']['num_layers'],
        num_actions=num_actions,
        dropout=config['state_updater']['dropout'],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config['training']['max_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['max_epochs']} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=config['training']['gradient_clip'],
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Logging
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2%}, "
              f"Baseline: {train_metrics['baseline_accuracy']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.2%}, "
              f"Baseline: {val_metrics['baseline_accuracy']:.2%}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/baseline_accuracy': train_metrics['baseline_accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/baseline_accuracy': val_metrics['baseline_accuracy'],
                'lr': scheduler.get_last_lr()[0],
            })

        # Checkpointing
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'config': config,
            }, os.path.join(output_dir, 'best_model.pt'))

            print(f"  -> New best model saved (val acc: {best_val_acc:.2%})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Final summary
    print("\n=== Training Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Majority baseline: {majority_baseline:.2%}")
    print(f"Improvement over baseline: {best_val_acc - majority_baseline:.2%}")

    # Go/no-go check
    if best_val_acc > majority_baseline + 0.05:  # At least 5% above majority
        print("\n[GO] Model shows meaningful improvement over baseline!")
    else:
        print("\n[NO-GO] Model does not sufficiently beat baseline. Debug needed.")

    if use_wandb:
        wandb.finish()

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectories JSONL")
    parser.add_argument("--output", type=str, default="checkpoints/phase1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        use_wandb=not args.no_wandb,
    )
