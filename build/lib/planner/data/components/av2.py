# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Argoverse 2 dataset API."""
from pathlib import Path
from typing import List, Literal, Optional

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import Dataset

from planner.data.components.av2_factories import (
    create_sceanrio_state_from_api,
)
from planner.data.dataclass import Scenario
from planner.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)


class AV2Dataset(Dataset):
    """Implementation of PyTorch ``Dataset`` for Argoverse 2 dataset."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        # transform: Optional[Transforms] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Initialize a new :class:`Argoverse2Dataset` instance.

        Args:
            root (str): Root directory of the `Argoverse 2` dataset.
            split (Literal["train", "val", "test"], optional): The dataset
                split to load. Defaults to `"train"`.
            transform (Optional[Transforms], optional): The transform to apply
                on the data. Defaults to `None`.
            radius (Optional[float], optional): The radius of observation
                range for map elements. Defaults to `None`.
        """
        # save hyperparameters
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.root = Path(root).resolve()
        self.radius = radius
        if self.radius is not None:
            assert self.radius >= 0, f"Invalid radius: {self.radius}"

        # check if raw data directory exists
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                "Raw data files not found or incomplete at "
                f"{str(self.raw_dir):s}. "
                "Please refer to the README for instructions on how to "
                "download and prepare the dataset."
            )

        # initialize the container for scenario IDs
        self._scenario_ids = [
            subdir.name for subdir in self.raw_dir.iterdir() if subdir.is_dir()
        ]
        LOGGER.info(f"Found {len(self._scenario_ids)} {self.split} scenarios.")

    def __getitem__(self, index: int) -> Scenario:
        scenario_id = self.scenario_ids[index]
        map_api = self.get_map_api(scenario_id)
        scenario_api = self.get_scenario_api(scenario_id)

        return create_sceanrio_state_from_api(
            scenario_id=str(scenario_id),
            map_api=map_api,
            scenario_api=scenario_api,
            radius=self.radius,
        )

    def __len__(self) -> int:
        return len(self.scenario_ids)

    @property
    def cache_dir(self) -> Path:
        """pathlib.Path: The cache directory."""
        return self.root.joinpath("cache", self.split)

    @property
    def raw_dir(self) -> Path:
        """pathlib.Path: The raw data directory."""
        return self.root.joinpath("raw", self.split)

    @property
    def scenario_ids(self) -> List[str]:
        """List[str]: The list of scenario IDs."""
        return self._scenario_ids

    def get_map_api(self, scenario_id: str) -> ArgoverseStaticMap:
        """Initialize the map API.

        Args:
            scenario_id (str): The scenario ID.

        Returns:
            ArgoverseStaticMap: The map API instance.
        """
        static_map_path = self.raw_dir.joinpath(
            scenario_id, f"log_map_archive_{scenario_id}.json"
        )
        map_api = ArgoverseStaticMap.from_json(static_map_path=static_map_path)
        return map_api

    def get_scenario_api(self, scenario_id: str) -> ArgoverseScenario:
        """Initialize the scenario API.

        Args:
            scenario_id (str): The scenario ID.

        Returns:
            ArgoverseScenario: The scenario API instance.
        """
        scenario_path = self.raw_dir.joinpath(
            scenario_id, f"scenario_{scenario_id}.parquet"
        )
        scenario_api = scenario_serialization.load_argoverse_scenario_parquet(
            scenario_path=scenario_path
        )
        return scenario_api
