import os
import pickle
import numpy as np
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ==========================
#  Configuración general
# ==========================

PKL_PATH = r"Modulo_ML\Modulo2_DeepL\dataset\ucf101_2d.pkl"
SPLIT_TRAIN = "train1"
SPLIT_TEST = "test1"
MAX_SEQ_LEN = 120          # número máximo de frames por video (ajustable)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 101          # UCF101 → 101 clases


# ==========================
#  Dataset
# ==========================

class UCF101SkeletonDataset(Dataset):
    """
    Dataset para UCF101 usando esqueletos 2D.
    Cada muestra devuelve:
        x: tensor [MAX_SEQ_LEN, feature_dim]
        mask: tensor [MAX_SEQ_LEN] con 1 donde hay datos válidos, 0 en padding
        label: entero [0..100]
    """
    def __init__(self, pkl_path: str, split_name: str, max_seq_len: int = 120):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Carga del .pkl
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        annotations: List[Dict[str, Any]] = data["annotations"]
        split: Dict[str, List[str]] = data["split"]

        # Mapeo de frame_dir → anotación
        self.by_name = {ann["frame_dir"]: ann for ann in annotations}

        # Lista de nombres de videos para el split
        self.video_names = split[split_name]

        # Checamos dimensión de features usando el primer ejemplo
        first_ann = self.by_name[self.video_names[0]]
        keypoint = first_ann["keypoint"]       # shape: [1, T, 17, 2]
        keypoint_score = first_ann["keypoint_score"]  # [1, T, 17]

        # Supondremos: features = [x, y, score] para cada joint
        _, T, J, _ = keypoint.shape
        self.num_joints = J
        self.feature_dim = J * 3  # (x, y, score) * 17

        print(f"[Dataset] Split: {split_name} | Videos: {len(self.video_names)}")
        print(f"[Dataset] Joints: {self.num_joints} | Feature dim/frame: {self.feature_dim}")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        name = self.video_names[idx]
        ann = self.by_name[name]

        keypoint = ann["keypoint"]        # [1, T, 17, 2]
        score = ann["keypoint_score"]     # [1, T, 17]
        label = ann["label"]

        # Tomamos solo la primera persona: [T, 17, 2] y [T, 17]
        kp = keypoint[0]                  # [T, 17, 2]
        sc = score[0]                     # [T, 17]

        T = kp.shape[0]

        # Normalización simple: escalamos por la resolución original
        # para tener valores aproximadamente entre 0 y 1
        # (opcional, puedes cambiar esto después)
        # original_shape = ann["original_shape"]  # (h, w)
        h, w = ann["original_shape"]
        kp_norm = np.empty_like(kp, dtype=np.float32)
        kp_norm[..., 0] = kp[..., 0] / w   # x / ancho
        kp_norm[..., 1] = kp[..., 1] / h   # y / alto

        # Concatenamos (x, y, score) por cada joint
        # kp_norm: [T, 17, 2] → reshape [T, 17*2]
        # sc:      [T, 17]    → reshape [T, 17]
        coords = kp_norm.reshape(T, -1)     # [T, 34]
        scores = sc.reshape(T, -1).astype(np.float32)  # [T, 17]
        features = np.concatenate([coords, scores], axis=-1)  # [T, 51]

        # Padding / truncado de la secuencia a MAX_SEQ_LEN
        if T >= self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - T
            pad = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
            mask = np.concatenate([np.ones(T, dtype=np.float32),
                                   np.zeros(pad_len, dtype=np.float32)], axis=0)

        x = torch.from_numpy(features)       # [MAX_SEQ_LEN, feature_dim]
        mask = torch.from_numpy(mask)        # [MAX_SEQ_LEN]
        y = torch.tensor(label, dtype=torch.long)

        return x, mask, y


# ==========================
#  Modelo: LSTM bidireccional
# ==========================

class ActionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 101,
                 bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x, mask=None):
        """
        x: [B, T, F]
        mask: [B, T] con 1 en frames válidos, 0 en padding
        """
        if mask is not None:
            lengths = mask.sum(dim=1).long()  # [B]
            # pack_padded_sequence requiere que esté en CPU
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, c_n) = self.lstm(packed)
        else:
            out, (h_n, c_n) = self.lstm(x)

        # h_n: [num_layers * num_directions, B, hidden_dim]
        if self.bidirectional:
            # concatenamos las últimas capas forward y backward
            h_forward = h_n[-2]   # [B, hidden_dim]
            h_backward = h_n[-1]  # [B, hidden_dim]
            h_last = torch.cat([h_forward, h_backward], dim=1)  # [B, 2*hidden_dim]
        else:
            h_last = h_n[-1]      # [B, hidden_dim]

        logits = self.fc(h_last)  # [B, num_classes]
        return logits


# ==========================
#  Funciones de entrenamiento
# ==========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for x, mask, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, mask, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(x, mask)
            loss = criterion(logits, y)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ==========================
#  Main
# ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar datasets
    train_dataset = UCF101SkeletonDataset(PKL_PATH, SPLIT_TRAIN, MAX_SEQ_LEN)
    test_dataset = UCF101SkeletonDataset(PKL_PATH, SPLIT_TEST, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Crear modelo
    input_dim = train_dataset.feature_dim
    model = ActionLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        bidirectional=True,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Comenzando entrenamiento...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"  Train  | loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"  Test   | loss: {val_loss:.4f} | acc: {val_acc:.4f}")

    # Guardar modelo
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "action_lstm_ucf101.pth"))
    print("Modelo guardado en checkpoints/action_lstm_ucf101.pth")


if __name__ == "__main__":
    main()
