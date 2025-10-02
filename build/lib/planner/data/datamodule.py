# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Unified data loading module for Neural Variational Agents."""
from typing import Callable, Optional, Sequence

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from planner.data.components.av2 import AV2Dataset
from planner.data.dataclass import Scenario
from planner.type import PathLike
from planner.utils.logging import get_logger

__all__ = ["NeVADataModule", "AV2DataModule"]

# Constants
LOGGER = get_logger(__name__)


class NeVADataModule(LightningDataModule):
    """Unified data loading module for Neural Variational Agents."""

    train_dataset: Optional[Dataset]
    """Optional[Dataset]: The training dataset."""
    val_dataset: Optional[Dataset]
    """Optional[Dataset]: The validation dataset."""
    test_dataset: Optional[Dataset]
    """Optional[Dataset]: The testing dataset."""

    def __init__(
        self,
        batch_size: int,
        collate_fn: Callable = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a new :class:`NeVADataModule` instance."""
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            LOGGER.warning("No training dataset initialized.")
            return None

        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            LOGGER.warning("No validation dataset initialized.")
            return None

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            LOGGER.warning("No testing dataset initialized.")
            return None

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class AV2DataModule(NeVADataModule):
    def __init__(
        self,
        batch_size: int,
        root: PathLike,
        num_workers: int = 0,
        pin_memory: bool = False,
        radius: Optional[float] = 50.0,
        train: bool = True,
        val: bool = True,
        test: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a new :class:`AV2DataModule` instance.

        Args:
            batch_size (int): The batch size for the data loader.
            root (PathLike): The root directory of the Argoverse 2 dataset.
            num_workers (int, optional): The number of workers for the data
                loader. Defaults to `0`.
            pin_memory (bool, optional): Whether to pin memory for the data
                loader. Defaults to `False`.
            radius (Optional[float], optional): The radius of observation
                range for map elements. Defaults to `50.0`.
            train (bool, optional): Whether to load the training dataset.
                Defaults to `True`.
            val (bool, optional): Whether to load the validation dataset.
                Defaults to `True`.
            test (bool, optional): Whether to load the testing dataset.
                Defaults to `False`.
        """
        # initialize the datasets
        print("RADIUS: ", radius)
        if train:
            self.train_dataset = AV2Dataset(
                root=root, split="train", radius=radius
            )
        else:
            self.train_dataset = None
        if val:
            self.val_dataset = AV2Dataset(
                root=root, split="val", radius=radius
            )
        else:
            self.val_dataset = None
        if test:
            self.test_dataset = AV2Dataset(
                root=root, split="test", radius=radius
            )
        else:
            self.test_dataset = None

        super().__init__(
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            *args,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch: Sequence[Scenario]) -> Scenario:
        """Collate a batch of Argoverse 2 scenarios into a single scenario.

        Args:
            batch (Sequence[Scenario]): A sequence of scenarios.

        Returns:
            Scenario: The collated scenario
        """
        from planner.data.dataclass import MapPoint, ObjectProperty, Trajectory

        # collate the object properties
        scenario_id = [s.scenario_id for s in batch]
        log_trajectory = Trajectory(
            x=torch.stack([s.log_trajectory.x for s in batch]),
            y=torch.stack([s.log_trajectory.y for s in batch]),
            z=torch.stack([s.log_trajectory.z for s in batch]),
            yaw=torch.stack([s.log_trajectory.yaw for s in batch]),
            velocity_x=torch.stack(
                [s.log_trajectory.velocity_x for s in batch]
            ),
            velocity_y=torch.stack(
                [s.log_trajectory.velocity_y for s in batch]
            ),
            timestamp_ms=torch.stack(
                [s.log_trajectory.timestamp_ms for s in batch]
            ),
            length=torch.stack([s.log_trajectory.length for s in batch]),
            width=torch.stack([s.log_trajectory.width for s in batch]),
            height=torch.stack([s.log_trajectory.height for s in batch]),
            observed=torch.stack([s.log_trajectory.observed for s in batch]),
            valid=torch.stack([s.log_trajectory.valid for s in batch]),
        )
        log_trajectory.validate()

        object_property = ObjectProperty(
            ids=torch.stack([s.object_property.ids for s in batch]),
            object_types=torch.stack(
                [s.object_property.object_types for s in batch]
            ),
            valid=torch.stack([s.object_property.valid for s in batch]),
            is_sdc=torch.stack([s.object_property.is_sdc for s in batch]),
            is_target=torch.stack(
                [s.object_property.is_target for s in batch]
            ),
        )
        object_property.validate()

        map_point = MapPoint(
            x=torch.stack([s.map_point.x for s in batch]),
            y=torch.stack([s.map_point.y for s in batch]),
            z=torch.stack([s.map_point.z for s in batch]),
            dir_x=torch.stack([s.map_point.dir_x for s in batch]),
            dir_y=torch.stack([s.map_point.dir_y for s in batch]),
            dir_z=torch.stack([s.map_point.dir_z for s in batch]),
            types=torch.stack([s.map_point.types for s in batch]),
            ids=torch.stack([s.map_point.ids for s in batch]),
            valid=torch.stack([s.map_point.valid for s in batch]),
        )
        map_point.validate()

        out = Scenario(
            scenario_id=scenario_id,
            log_trajectory=log_trajectory,
            object_property=object_property,
            map_point=map_point,
            current_time_step=batch[0].current_time_step,
        )
        out.validate()

        return out
