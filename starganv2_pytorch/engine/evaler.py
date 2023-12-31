# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from stargan_pytorch.data.celeba import CelebADataset
from stargan_pytorch.data.prefetcher import CPUPrefetcher, CUDAPrefetcher
from stargan_pytorch.models.stargan import generator
from stargan_pytorch.models.utils import load_state_dict
from stargan_pytorch.utils import create_labels, denorm


class Evaler:
    def __init__(
            self,
            config: Dict,
            device: torch.device,
            save_dir: Union[str, Path],
    ) -> None:
        self.config = config
        self.device = device
        self.save_dir = save_dir

        self.model = self.build_model()
        self.dataloader = self.load_datasets()
        self.batches = len(self.dataloader)

        self.load_checkpoint()

        # Disable gradient calculation
        torch.set_grad_enabled(False)

    def build_model(self) -> nn.Module:
        """Build the generator, discriminator and exponential average generator models

        Returns:
            nn.Module: generator, discriminator and exponential average generator models
        """

        model = generator(c_dim=self.config["MODEL"]["G"]["C_DIM"])
        model = model.to(self.device)
        return model

    def load_datasets(self) -> CPUPrefetcher or CUDAPrefetcher:
        """Load the dataset and generate the dataset iterator

        Returns:
            DataLoader: dataset iterator
        """

        transform = transforms.Compose([
            transforms.CenterCrop(self.config["DATASETS"]["CROP_IMG_SIZE"]),
            transforms.Resize(self.config["DATASETS"]["RESIZE_IMG_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Load the train dataset
        datasets = CelebADataset(
            self.config["DATASETS"]["IMGS_DIR"],
            transform,
            self.config["DATASETS"]["ATTR_PATH"],
            self.config["DATASETS"]["SELECTED_ATTRS"],
        )
        # generate dataset iterator
        dataloader = DataLoader(datasets,
                                batch_size=self.config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=True)

        if self.device == "cuda":
            # Replace the data set iterator with CUDA to speed up
            dataloader = CUDAPrefetcher(dataloader, self.device)
        else:
            dataloader = CPUPrefetcher(dataloader)

        return dataloader

    def load_checkpoint(self) -> None:
        """Load the checkpoint"""

        model_weights_path = self.config["TEST"]["CHECKPOINT"]["MODEL_WEIGHTS_PATH"]
        print(f"Load checkpoint from '{model_weights_path}'")
        state_dict = torch.load(model_weights_path, map_location=self.device)["state_dict"]
        self.model = load_state_dict(self.model, state_dict, False)

    def eval(self):
        # Initialize data batches
        batch_index = 0
        # Set the dataset iterator pointer to 0
        self.dataloader.reset()
        # Record the start time of training a batch
        end = time.time()
        # load the first batch of data
        batch_data = self.dataloader.next()

        while batch_data is not None:
            # Load batches of data
            imgs = batch_data["img"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True)

            # Prepare input images and target domain labels.
            labels_list = create_labels(labels,
                                        self.config["MODEL"]["G"]["C_DIM"],
                                        "CelebA",  # Only support CelebA dataset
                                        self.config["DATASETS"]["SELECTED_ATTRS"])

            # Translate images.
            imgs_fake_list = [imgs]
            for labels in labels_list:
                imgs_fake_list.append(self.model(imgs, labels))

            # Save the translated images.
            imgs_concat = torch.cat(imgs_fake_list, dim=3)
            sample_path = os.path.join(self.save_dir, f"{batch_index:06d}.jpg")
            save_image(denorm(imgs_concat.data.cpu()), sample_path, nrow=1, padding=0)

            # Preload the next batch of data
            batch_data = self.dataloader.next()
            # Record the end time of training a batch
            batch_index += 1
