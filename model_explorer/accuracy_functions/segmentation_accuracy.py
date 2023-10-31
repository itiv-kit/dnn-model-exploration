import torch
import tqdm
import numpy as np


# Taken from the Deeplabv3 repository and modified for us
class StreamSegMetrics():
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

    def __str__(self):
        return f"Overall Acc: {self.overall_acc:.2f}, Mean IoU: {self.mean_iou:.2f}"

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    @property
    def overall_acc(self) -> float:
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    @property
    def mean_acc(self) -> float:
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)

    @property
    def mean_iou(self) -> float:
        iu = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) +
                                               self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix))
        return np.nanmean(iu)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def compute_sematic_segmentation_accuracy(base_model, dataloader_generator, progress=True, title="", **kwargs) -> float:
    """Compute sematic segmentation accuracy. Currently the function only
    returns the accuracy, however, it might be changed to whatever objective the
    user has.

    Args:
        base_model (nn.Module): input Base model for evaluation
        dataloader_generator: dataset loader
        progress (bool, optional): Show progress? Defaults to True.
        title (str, optional): Given title of the progress bar. Defaults to "".

    Returns:
        float: Algorithm pixelwise accuracy
    """
    dev_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_string)

    dataset_size = len(dataloader_generator)
    dataloader = dataloader_generator.get_dataloader()

    progress_bar = tqdm.tqdm(total=dataset_size, desc=title, position=0, disable=not progress)

    n_classes = kwargs.get('n_classes')
    crop_range = kwargs.get('crop_range', None)

    current_metrics = StreamSegMetrics(n_classes=n_classes)  # FIXME, inferrr ...

    model = base_model.to(device)

    model.eval()
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)

            y_prob = model(x)
            y_pred = y_prob.detach().max(dim=1)[1].cpu().numpy()  # max[1] to get indices
            if crop_range:
                y_pred = y_pred[:, crop_range[0]:crop_range[1], :]
            y_true = target.cpu().numpy()

            current_metrics.update(y_true, y_pred)

            progress_bar.update(len(y_pred))

    return current_metrics.overall_acc
