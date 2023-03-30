import torch
import tqdm


def compute_pixelwise_segmentation_accuracy(base_model, dataloader_generator, progress=True, title="") -> float:
    dev_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_string)
    cpu_device = torch.device("cpu")

    dataset_size = len(dataloader_generator)
    dataloader = dataloader_generator.get_dataloader()

    progress_bar = tqdm.tqdm(total=dataset_size, ascii=True, desc=title, position=0, disable=not progress)

    model = base_model.to(device)

    running_pixel_acc = []

    model.eval()
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            y_prob = model(x)
            y_pred = y_prob.argmax(1)[4:-4, :]

            pixel_acc = (target == y_pred).float().mean()
            running_pixel_acc.append(pixel_acc)

            progress_bar.update(y_pred.size(0))

    pixel_accs = torch.stack(running_pixel_acc)

    return float(pixel_accs.float().mean().to(cpu_device))


accuracy_function = compute_pixelwise_segmentation_accuracy
