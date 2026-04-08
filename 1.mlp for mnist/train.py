import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from src.config import CONFIG,device
from src.data import train_dataloader,val_dataloader
from src.model import model


#printing to ensure everything looks good
print(f"Using device: {device}")
print(CONFIG)
print(f"Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)

#====== wandb set up ======
wandb.init(
    project = 'mnist',
    config=CONFIG,
    name='mlp_baseline'
)

# Watch model gradients
wandb.watch(model,log='gradients',log_freq=100)

#====== training and evaluation mechanism =======
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_dataloader):
        # 1. Move to device
        images,labels=images.to(device),labels.to(device)
        # 2. Zero gradients
        optimizer.zero_grad()
        # 3. Forward pass
        logits=model(images)
        # 4. Compute loss
        loss = criterion(logits,labels)
        # 5. Backward pass
        loss.backward()
        # 6. Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        # 7. Optimizer step
        optimizer.step()
        # 8. Accumulate stats
        # Hint: total_loss += loss.item() * images.size(0)  ← why multiply? to undo the mean
        # Hint: preds = logits.argmax(dim=1)
        total_loss += loss.item()*images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += images.size(0)

        # 9. Log to W&B every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/step': epoch * len(train_dataloader) + batch_idx
            })

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            # 1. Move to device
            images,labels = images.to(device),labels.to(device)
            # 2. Forward pass
            logits = model(images)
            # 3. Compute loss
            loss = criterion(logits,labels)
            # 4. Accumulate stats
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds==labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

#======= start training and get results =======
best_val_acc = 0.0

for epoch in range(1, CONFIG['epochs'] + 1):
    # 1. Train
    train_loss,train_acc = train_one_epoch(epoch)
    # 2. Validate
    val_loss,val_acc = evaluate()
    # 3. Step scheduler
    scheduler.step()
    # 4. Log to W&B
    wandb.log({
        'epoch':          epoch,
        "train/loss":     train_loss,
        "train/accuracy": train_acc,
        "val/loss":       val_loss,
        'lr':             scheduler.get_last_lr()[0]
        # ... fill in the rest
    })

    # 5. Print progress
    print(
        f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
        # ... fill in train/val metrics
        f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
    )

    # 6. Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  ✓ New best val accuracy: {best_val_acc:.4f} — model saved")


# Finish
print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
wandb.summary['best_val_accuracy'] = best_val_acc
wandb.finish()
