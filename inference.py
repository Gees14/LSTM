# inference.py

import os
import torch

# Importamos cosas definidas en main.py
from main import (
    UCF101SkeletonDataset,
    ActionLSTM,
    ActionMLP,
    PKL_PATH,
    SPLIT_TEST,
    MAX_SEQ_LEN,
    NUM_CLASSES,
    MODEL_TYPE,
)


# ==========================
#  Inferencia sobre ejemplos individuales
# ==========================

def load_trained_model(checkpoint_path, pkl_path=PKL_PATH, split_name=SPLIT_TEST):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset de prueba (para obtener feature_dim y datos)
    test_dataset = UCF101SkeletonDataset(pkl_path, split_name, MAX_SEQ_LEN)

    if MODEL_TYPE == "lstm":
        model = ActionLSTM(
            input_dim=test_dataset.feature_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=NUM_CLASSES,
            bidirectional=True,
            dropout=0.3
        ).to(device)
    elif MODEL_TYPE == "mlp":
        model = ActionMLP(
            input_dim=test_dataset.feature_dim,
            hidden_dim=128,
            num_classes=NUM_CLASSES,
            dropout=0.3
        ).to(device)
    else:
        raise ValueError(f"MODEL_TYPE desconocido: {MODEL_TYPE}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, test_dataset, device


def predict_sample(model, dataset, device, idx=0, topk=5):
    """
    Predice la acción para un video del dataset de test.
    idx: índice del video (0..len-1)
    topk: cuántas clases mostrar (Top-k)
    """
    x, mask, y_true = dataset[idx]

    x = x.unsqueeze(0).to(device)      # [1, T, F]
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x, mask)        # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0]  # [num_classes]

    # Top-k
    top_probs, top_indices = torch.topk(probs, k=topk)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    true_label = y_true.item()
    true_name = dataset.get_class_name(true_label)

    print(f"\nVideo idx: {idx}")
    print(f"Ground truth: {true_label} ({true_name})")
    print(f"Top-{topk} predicciones:")

    for rank, (cls, p) in enumerate(zip(top_indices, top_probs), start=1):
        cls = int(cls)
        cls_name = dataset.get_class_name(cls)
        print(f"  {rank}) clase {cls} ({cls_name})  prob={p:.3f}")


# ==========================

if __name__ == "__main__":
    # Armamos el nombre del checkpoint según el tipo de modelo
    # Debe coincidir con cómo lo guarda main.py
    ckpt_name = f"action_{MODEL_TYPE}_ucf101.pth"
    checkpoint_path = os.path.join(
        r"Modulo2_DeepL\checkpoints", ckpt_name
    )

    print(f"Usando MODEL_TYPE = {MODEL_TYPE}")
    print(f"Cargando checkpoint desde: {checkpoint_path}")

    model, test_dataset, device = load_trained_model(checkpoint_path)

    # Probar con algunos ejemplos
    predict_sample(model, test_dataset, device, idx=0, topk=5)
    predict_sample(model, test_dataset, device, idx=10, topk=5)
    predict_sample(model, test_dataset, device, idx=25, topk=5)
