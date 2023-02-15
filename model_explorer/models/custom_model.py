import torch

from tqdm import tqdm
from torch import nn

from model_explorer.utils.data_loader_generator import DataLoaderGenerator
from model_explorer.utils.logger import logger


class CustomModel():
    """Base Class for models in which modules can be replaced with explorable modules.
    """

    def __init__(self,
                 base_model: nn.Module,
                 device: torch.device,
                 verbose: bool = False) -> None:
        super().__init__()

        self.base_model = base_model
        self._bit_widths = {}
        self.device = device
        self.verbose = verbose

        self.explorable_modules = []
        self.explorable_module_names = []

        # Training things
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=1,
                                                            gamma=0.1)

    def get_explorable_parameter_count(self) -> int:
        return len(self.explorable_modules)

    def load_parameters(self, filename: str):
        self.base_model.load_state_dict(
            torch.load(filename, map_location=self.device))

    def save_parameters(self, filename: str):
        torch.save(self.base_model.state_dict(), filename)

    def retrain(self,
                train_dataloader_generator: DataLoaderGenerator,
                test_dataloader_generator: DataLoaderGenerator,
                accuracy_function: callable,
                num_epochs: int = 10,
                progress: bool = False) -> list:
        self.base_model.to(self.device)

        epoch_accs = []

        for epoch_idx in range(num_epochs):
            self.base_model.train()

            logger.info("Starting Epoch {} / {}".format(epoch_idx + 1, num_epochs))

            if progress:
                pbar = tqdm(total=len(train_dataloader_generator),
                            desc="Epoch {} / {}".format(epoch_idx + 1, num_epochs),
                            position=1)

            running_loss = 0.0
            train_dataloader = train_dataloader_generator.get_dataloader()

            for image, target, *_ in train_dataloader:
                image, target = image.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):
                    output = self.base_model(image)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * output.size(0)

                    if progress:
                        pbar.update(output.size(0))

            self.lr_scheduler.step()

            if progress:
                pbar.close()

            epoch_loss = running_loss / len(train_dataloader_generator)
            logger.info("Ran Epoch {} / {} with loss of: {}".format(epoch_idx + 1, num_epochs, epoch_loss))

            self.base_model.eval()
            # FIXME!
            acc = accuracy_function(self.base_model,
                                    test_dataloader_generator,
                                    progress,
                                    title="Eval {} / {}".format(epoch_idx + 1, num_epochs))
            epoch_accs.append(acc)
            logger.info("Inference Accuracy after Epoch {}: {}".format(epoch_idx + 1, acc))

        return epoch_accs
