"""
Phase 2B Training: Train the state-conditioned retriever.

This script trains the query_proj layer to map state vectors to the
chunk embedding space, so that states retrieve relevant chunks.

Key approach:
- Load Phase 1 checkpoint (encoder + GRU)
- Freeze encoder and GRU
- Train only the retriever's query_proj
- Use contrastive loss with click-target heuristics
"""

import os
import argparse
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.retrieval_dataset import RetrievalDataset, collate_retrieval_batch
from models.encoder import EventEncoder
from models.state_updater import GRUStateUpdater
from models.retriever import StateConditionedRetriever


class RetrievalModel(nn.Module):
    """
    Combined model for retrieval training.

    Consists of:
    - EventEncoder (frozen)
    - GRUStateUpdater (frozen)
    - StateConditionedRetriever (trainable query_proj)
    """

    def __init__(
        self,
        encoder: EventEncoder,
        state_updater: GRUStateUpdater,
        retriever: StateConditionedRetriever,
        freeze_encoder: bool = True,
        freeze_state_updater: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.state_updater = state_updater
        self.retriever = retriever

        # Freeze components as specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if freeze_state_updater:
            for param in self.state_updater.parameters():
                param.requires_grad = False

    def compute_state(self, events):
        """Compute state from event history."""
        if len(events) == 0:
            h = self.state_updater.init_state(1)
            return self.state_updater.get_state_vector(h).squeeze(0)

        event_embs = self.encoder.encode_sequence(events)
        event_embs = event_embs.unsqueeze(0)
        outputs, _ = self.state_updater(event_embs)
        return outputs[0, -1]  # Final state

    def forward(self, events, chunks, pos_idx, neg_indices):
        """
        Forward pass for one sample.

        Args:
            events: List of (event_type, text) tuples
            chunks: List of chunk strings
            pos_idx: Index of positive chunk
            neg_indices: List of negative chunk indices

        Returns:
            loss: Contrastive loss for this sample
            metrics: Dict with pos_score, neg_scores, etc.
        """
        # Compute state from events
        state = self.compute_state(events)

        # Embed all relevant chunks
        all_indices = [pos_idx] + neg_indices
        relevant_chunks = [chunks[i] for i in all_indices]
        chunk_embs = self.encoder.encode_texts(relevant_chunks)

        pos_emb = chunk_embs[0]  # First is positive
        neg_embs = chunk_embs[1:]  # Rest are negatives

        # Compute loss using retriever's method
        loss = self.retriever.compute_retrieval_loss(
            state.unsqueeze(0),
            pos_emb.unsqueeze(0),
            neg_embs.unsqueeze(0),
            margin=0.2,
        )

        # Compute metrics
        with torch.no_grad():
            query = self.retriever.query_proj(state)
            query = F.normalize(query, dim=-1)
            pos_emb_norm = F.normalize(pos_emb, dim=-1)
            neg_embs_norm = F.normalize(neg_embs, dim=-1)

            pos_score = (query * pos_emb_norm).sum()
            neg_scores = torch.matmul(query, neg_embs_norm.T)

            # Is positive ranked higher than all negatives?
            rank_correct = (pos_score > neg_scores).all().float()

        metrics = {
            'pos_score': pos_score.item(),
            'neg_score_mean': neg_scores.mean().item(),
            'neg_score_max': neg_scores.max().item(),
            'rank_correct': rank_correct.item(),
        }

        return loss, metrics


def train_epoch(
    model: RetrievalModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    # But keep frozen parts in eval mode
    model.encoder.eval()
    model.state_updater.eval()

    total_loss = 0
    total_rank_correct = 0
    total_pos_score = 0
    total_neg_score = 0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        batch_loss = 0
        batch_metrics = {'rank_correct': 0, 'pos_score': 0, 'neg_score_mean': 0}

        # Process each sample in batch
        for i in range(len(batch['events'])):
            events = batch['events'][i]
            chunks = batch['chunks'][i]
            pos_idx = batch['pos_indices'][i]
            neg_indices = batch['neg_indices'][i]

            loss, metrics = model(events, chunks, pos_idx, neg_indices)
            batch_loss += loss

            batch_metrics['rank_correct'] += metrics['rank_correct']
            batch_metrics['pos_score'] += metrics['pos_score']
            batch_metrics['neg_score_mean'] += metrics['neg_score_mean']

        # Average loss over batch
        batch_loss = batch_loss / len(batch['events'])

        # Backward pass
        batch_loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.retriever.parameters(), gradient_clip)

        optimizer.step()

        # Update metrics
        batch_size = len(batch['events'])
        total_loss += batch_loss.item() * batch_size
        total_rank_correct += batch_metrics['rank_correct']
        total_pos_score += batch_metrics['pos_score']
        total_neg_score += batch_metrics['neg_score_mean']
        num_samples += batch_size

        pbar.set_postfix({
            'loss': f'{batch_loss.item():.3f}',
            'rank_acc': f'{batch_metrics["rank_correct"]/batch_size:.2%}',
        })

    return {
        'loss': total_loss / num_samples,
        'rank_accuracy': total_rank_correct / num_samples,
        'pos_score': total_pos_score / num_samples,
        'neg_score': total_neg_score / num_samples,
    }


@torch.no_grad()
def evaluate(
    model: RetrievalModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    total_rank_correct = 0
    total_pos_score = 0
    total_neg_score = 0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        for i in range(len(batch['events'])):
            events = batch['events'][i]
            chunks = batch['chunks'][i]
            pos_idx = batch['pos_indices'][i]
            neg_indices = batch['neg_indices'][i]

            loss, metrics = model(events, chunks, pos_idx, neg_indices)

            total_loss += loss.item()
            total_rank_correct += metrics['rank_correct']
            total_pos_score += metrics['pos_score']
            total_neg_score += metrics['neg_score_mean']
            num_samples += 1

    return {
        'loss': total_loss / num_samples,
        'rank_accuracy': total_rank_correct / num_samples,
        'pos_score': total_pos_score / num_samples,
        'neg_score': total_neg_score / num_samples,
    }


def train(
    config_path: str,
    data_path: str,
    phase1_checkpoint: str,
    output_dir: str,
):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config.get('device', 'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load Phase 1 checkpoint
    print(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
    checkpoint = torch.load(phase1_checkpoint, map_location=device)
    phase1_config = checkpoint.get('config', {})

    # Initialize encoder and state updater
    encoder = EventEncoder(
        text_model=phase1_config.get('encoder', {}).get('text_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        d_event=phase1_config.get('encoder', {}).get('d_event', 256),
    ).to(device)

    state_updater = GRUStateUpdater(
        d_event=phase1_config.get('encoder', {}).get('d_event', 256),
        d_state=phase1_config.get('state_updater', {}).get('d_state', 512),
        num_layers=phase1_config.get('state_updater', {}).get('num_layers', 2),
    ).to(device)

    # Load Phase 1 weights
    state_dict = checkpoint['model_state_dict']

    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state, strict=False)

    updater_state = {k.replace('state_updater.', ''): v for k, v in state_dict.items() if k.startswith('state_updater.')}
    state_updater.load_state_dict(updater_state, strict=False)

    print("Loaded encoder and state updater from Phase 1")

    # Initialize retriever (randomly initialized query_proj)
    retriever = StateConditionedRetriever(
        d_state=phase1_config.get('state_updater', {}).get('d_state', 512),
        d_chunk=config.get('retriever', {}).get('d_chunk', 384),
    ).to(device)

    # Create combined model
    phase2_config = config.get('phase2', {})
    model = RetrievalModel(
        encoder=encoder,
        state_updater=state_updater,
        retriever=retriever,
        freeze_encoder=True,
        freeze_state_updater=phase2_config.get('freeze_state_model', True),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    full_dataset = RetrievalDataset(
        trajectories_path=data_path,
        max_episodes=config.get('max_episodes', None),
        num_negatives=phase2_config.get('num_negatives', 8),
    )

    # Train/val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_retrieval_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_retrieval_batch,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Optimizer - only train retriever parameters
    optimizer = optim.AdamW(
        model.retriever.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
    )

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config['training']['max_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['max_epochs']} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            gradient_clip=config['training']['gradient_clip'],
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Logging
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Rank Acc: {train_metrics['rank_accuracy']:.2%}, "
              f"Pos: {train_metrics['pos_score']:.3f}, "
              f"Neg: {train_metrics['neg_score']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Rank Acc: {val_metrics['rank_accuracy']:.2%}, "
              f"Pos: {val_metrics['pos_score']:.3f}, "
              f"Neg: {val_metrics['neg_score']:.3f}")

        # Checkpointing
        if val_metrics['rank_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['rank_accuracy']
            patience_counter = 0

            os.makedirs(output_dir, exist_ok=True)

            # Save full model (encoder + state_updater + retriever)
            save_dict = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'state_updater_state_dict': state_updater.state_dict(),
                'retriever_state_dict': retriever.state_dict(),
                'val_rank_accuracy': best_val_acc,
                'config': config,
                'phase1_config': phase1_config,
            }
            torch.save(save_dict, os.path.join(output_dir, 'best_model.pt'))

            print(f"  -> New best model saved (rank acc: {best_val_acc:.2%})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Final summary
    print("\n=== Training Complete ===")
    print(f"Best validation rank accuracy: {best_val_acc:.2%}")

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectories JSONL")
    parser.add_argument("--phase1", type=str, required=True, help="Path to Phase 1 checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/phase2")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data,
        phase1_checkpoint=args.phase1,
        output_dir=args.output,
    )
