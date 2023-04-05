import tqdm
import torch


def compute_classification_accuracy(base_model, dataloader_generator, progress=True, title="") -> float:
    """Calculates the classification accuracy of the provided base classification model on the provided dataloader.
    The accuracy is calculated as the number of correct predictions by the number of samples.

    Args:
        base_model (nn.Model): The base classification model to be evaluated.
        dataloader (data.Dataloader):  The dataloader with the evaluation data
        progress (bool, optional): Wether to show a progress bar. Defaults to True.

    Returns:
        float: The accuracy of the provided base model on the provided dataloader.
    """
    dev_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_string)
    cpu_device = torch.device("cpu")

    dataset_size = len(dataloader_generator)
    dataloader = dataloader_generator.get_dataloader()

    progress_bar = tqdm.tqdm(total=dataset_size, ascii=True, desc=title, position=0, disable=not progress)

    model = base_model
    model = model.to(device)

    correct_pred = 0

    model.eval()
    with torch.no_grad():
        for X, y_true in dataloader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            correct_pred += (predicted_labels == y_true).sum()

            progress_bar.update(y_true.size(0))

    correct_pred = correct_pred.to(cpu_device)
    return correct_pred.float() / dataset_size


