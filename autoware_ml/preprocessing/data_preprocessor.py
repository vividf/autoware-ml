from typing import Sequence

from torch import nn

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTBatch


class DataPreprocessor:
    """Class for runtime preprocessing of multi-task data.

    This class is responsible for applying runtime preprocessing to the input data before it is fed into the model. It can be used to perform any necessary transformations or augmentations on the input data.

    Args:
        preprocessor_modules: A sequence of nn.Module instances that perform preprocessing
            on the input batch.
    """

    def __init__(self, preprocessor_modules: Sequence[nn.Module]) -> None:
        self.preprocessor_modules = preprocessor_modules

    def __call__(
        self, multi_task_gt_batch: MultiTaskGTBatch, is_training: bool
    ) -> MultiTaskBatchInputs:
        """Apply runtime preprocessing to the input batch.

        Args:
            multi_task_gt_batch (MultiTaskGTBatch): The input batch of data to be preprocessed.
            is_training (bool): Set True if DataPreprocessor is run in the training mode.

        Returns:
            MultiTaskBatchInputs: The batch of data after running the list of preprocessor_modules.
        """
        # Build a MultiTaskFeatures instance from the input batch
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch,
            voxels_data=None,  # Placeholder for voxelization
        )
        for module in self.preprocessor_modules:
            multi_task_batch_inputs = module(
                multi_task_batch_inputs=multi_task_batch_inputs,
                is_training=is_training,
            )
        return multi_task_batch_inputs
