import torch

from model_explorer.utils.logger import logger

from model_explorer.third_party.yolop_det_seg.lib.core.function import validate
from model_explorer.third_party.yolop_det_seg.lib.config import cfg
from model_explorer.third_party.yolop_det_seg.lib.core.loss import get_loss


def compute_yolop_combined_metric(base_model, dataloader_generator) -> list:
    """Compute all metrics for YOLOP and retrun a given set of them

    Args:
        base_model (nn.Module): Evaluation model, should be YOLOP model
        dataloader_generator: Dataset for evaluation

    Returns:
        list: List of results, currently containing lane line accuracy, drivable
        area mean IoU and detection MAP50
    """
    dev_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_string)

    model = base_model.to(device)

    validation_loader = dataloader_generator.dataloader
    criterion = get_loss(cfg, device)
    output_dir = 'results'

    cfg.TEST.PLOTS = False

    da_segmentation_results, ll_segmentation_results, detection_results, total_loss, _, timing = \
        validate(epoch=0,
                 config=cfg,
                 val_loader=validation_loader,
                 val_dataset=None,
                 model=model,
                 criterion=criterion,
                 output_dir=output_dir,
                 tb_log_dir=None,
                 writer_dict=None,
                 logger=logger,
                 device=dev_string)

    logger.debug(f"Results: da line acc: {da_segmentation_results[0]:.4f}, da iou: {da_segmentation_results[1]:.4f}, da miou: {da_segmentation_results[2]:.4f}")
    logger.debug(f"         ll line acc: {ll_segmentation_results[0]:.4f}, ll iou: {ll_segmentation_results[1]:.4f}, ll miou: {ll_segmentation_results[2]:.4f}")
    logger.debug(f"              det mp: {detection_results[0]:.4f},     mr: {detection_results[1]:.4f},   map50: {detection_results[2]:.4f}, map: {detection_results[3]:.4f}")

    return [ll_segmentation_results[0], da_segmentation_results[2], detection_results[2]]


accuracy_function = compute_yolop_combined_metric


