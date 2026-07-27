import torch
import pandas as pd
from tqdm.auto import tqdm

from classifier_weather.src.model import create_model
from classifier_weather.src.dataset import build_idx_to_target

def predict_ensemble(model_paths, test_dataloader, num_classes=3, device=None):
    all_probs = None
    filenames = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for path in tqdm(model_paths):
        model = create_model(num_classes=3, freeze_backbone=True, unfreeze_last_block=True, device=device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

        probs_list = []
        labels_list = []

        with torch.no_grad():
            for X_batch, labels in tqdm(test_dataloader):
                outputs = model(X_batch.to(device))
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())
                labels_list.extend(labels)

        probs_list = torch.cat(probs_list, dim=0)

        if all_probs is None:
            all_probs = probs_list
            filenames = labels_list
        else:
            all_probs += probs_list

    all_probs /= len(model_paths)
    final_preds = torch.argmax(all_probs, dim=1)

    return filenames, final_preds

def save_predict(base_dataset, predictions, filenames, output_path='../results/predictions_ensemble.csv'):
    class_to_target = {'rain': 0, 'fog': 1, 'snow': 2}
    model_idx_to_target = {
        base_dataset.class_to_idx[k]: v for k, v in class_to_target.items()
    }

    pred_labels = [model_idx_to_target[int(p)] for p in predictions]

    results_df = pd.DataFrame({'id': filenames, 'label': pred_labels})
    results_df.to_csv(output_path, index=False)