# inference.py

import torch

# Importamos cosas definidas en main.py
from main import (
    UCF101SkeletonDataset,
    ActionLSTM,
    PKL_PATH,
    SPLIT_TEST,
    MAX_SEQ_LEN,
    NUM_CLASSES,
)

# ==========================
#  Inferencia sobre ejemplos individuales
# ==========================

def load_trained_model(checkpoint_path, pkl_path=PKL_PATH, split_name=SPLIT_TEST):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset de prueba (para obtener feature_dim y datos)
    test_dataset = UCF101SkeletonDataset(pkl_path, split_name, MAX_SEQ_LEN)

    model = ActionLSTM(
        input_dim=test_dataset.feature_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        bidirectional=True,
        dropout=0.3
    ).to(device)

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

    print(f"Video idx: {idx}")
    print(f"Ground truth label (entero): {y_true.item()}")
    print("Top-{} predicciones:".format(topk))
    for rank, (cls, p) in enumerate(zip(top_indices, top_probs), start=1):
        print(f"  {rank}) clase {int(cls)}  prob={p:.3f}")


if __name__ == "__main__":
    checkpoint_path = r"Modulo_ML\Modulo2_DeepL\checkpoints\action_lstm_ucf101.pth"

    model, test_dataset, device = load_trained_model(checkpoint_path)

    # Probar con algunos ejemplos
    predict_sample(model, test_dataset, device, idx=0, topk=5)
    predict_sample(model, test_dataset, device, idx=10, topk=5)

    predict_sample(model, test_dataset, device, idx=25, topk=5)
