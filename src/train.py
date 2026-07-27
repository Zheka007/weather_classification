import torch.cuda
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets
import gc
import matplotlib.pyplot as plt
import os

from classifier_weather.src.model import create_model

device = "cuda" if torch.cuda.is_available() else 'cpu'

def plot_kfold_summary(fold_scores, save_path=None):
    """Bar chart итогового val F1 по каждому fold — для README."""
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [f'Fold {i+1}' for i in range(len(fold_scores))]

    bars = ax.bar(folds, fold_scores, color='#4C72B0')
    ax.axhline(np.mean(fold_scores), color='red', linestyle='--',
               label=f'Среднее: {np.mean(fold_scores):.4f}')

    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.005,
                f'{score:.3f}', ha='center')

    ax.set_ylabel('F1 score')
    ax.set_title('Val F1 по фолдам')
    ax.set_ylim(0, 1.0)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    plt.show()

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()

    loss_item = 0

    y_true = []
    y_pred = []

    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_item += loss.item()

        preds = outputs.argmax(dim=1)

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    loss_epoch = loss_item / len(dataloader)
    score = f1_score(y_true, y_pred, average='macro')

    return loss_epoch, score

def train(model, dataloader, val_dataloader, optimizer, criterion, num_epoch=30):

    train_losses = []
    train_scores = []
    val_scores = []

    for epoch in tqdm(range(num_epoch)):
        loss_epoch, score = train_one_epoch(model, dataloader, optimizer, criterion)
        print(f"Epoch: {epoch}, Loss: {loss_epoch:.4f}, F1: {score:.4f}")
        f1_val = val(model, val_dataloader)

        train_losses.append(loss_epoch)
        train_scores.append(score)
        val_scores.append(f1_val)

    return model, train_losses, train_scores, val_scores


def train_with_kfold(TRAIN_DATA_DIR, data_transforms, device,
                     n_splits=5, batch_size=32, num_epoch=15,
                     models_dir='models', results_dir='results'):

    base_dataset = datasets.ImageFolder(TRAIN_DATA_DIR)
    targets = np.array([label for _, label in base_dataset.samples])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['train']), train_idx)
        val_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['val']), val_idx)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = create_model(num_classes=3, freeze_backbone=True, unfreeze_last_block=True, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
            {'params': model.fc.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4}
        ])

        model, train_losses, train_scores, val_scores = train(
            model, train_dataloader, val_dataloader, optimizer, criterion, num_epoch=num_epoch
        )

        score = val_scores[-1]
        fold_scores.append(score)
        print(f"Fold: {fold+1}", f"F1: {score:.4f}")

        torch.save(model.state_dict(), f'../{models_dir}/model_fold_{fold}.pth')

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nСредний F1 по фолдам: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    plot_kfold_summary(fold_scores, save_path=f'../{results_dir}/kfold_summary.png')

    return fold_scores

def val(model, dataloader):

    model.eval()

    with torch.no_grad():

        y_pred = []
        y_true = []

        for X_batch, y_batch in tqdm(dataloader):
            output = model(X_batch.to(device))
            preds = output.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.numpy())

        score = f1_score(y_true, y_pred, average='macro')
    print("Validation:")
    print(f"F1: {score:.2f}")
    return score