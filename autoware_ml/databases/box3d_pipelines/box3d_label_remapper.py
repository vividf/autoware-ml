from types import MappingProxyType
from typing import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel


class Box3DLabelRemapper(Box3DPipeline):
    """
    Pipeline to remap the label names and indices of the 3D bounding boxes to another label name.
    If the new label name for the box3d is not in the target class names, it will map to the ignore label index.
    """

    def __init__(
        self,
        label_remapper: MappingProxyType[str, str],
        label_names: Sequence[str],
        ignore_label_index: int,
    ):
        """
        Initialize Box3DLabelRemapper.

        Args:
          label_remapper: Mapping to remap label names.
          label_names: List of label names in the database, used for category mapping.
          ignore_label_index: Index to use for ignored labels.
        """
        super().__init__()
        self.label_remapper = label_remapper
        self.label_names = label_names
        self.ignore_label_index = ignore_label_index
        self.label_index_remapper = {
            label_name: index for index, label_name in enumerate(label_names)
        }

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return (
            f"{self.__class__.__name__}(label_remapper={self.label_remapper}, "
            f"label_names={self.label_names}, "
            f"ignore_label_index={self.ignore_label_index})"
        )

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Remap the label names of the 3D bounding boxes to another label name.

        The current label name is remapped, not the dataset label name, so that a pipeline can
        run several remappers in a row: every pass builds on the label name the previous pass
        assigned, and a box no pass remaps keeps the dataset label name it started with.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            if box3d_data_model.box3d_label_name in self.label_remapper:
                new_box3d_label_name = self.label_remapper[box3d_data_model.box3d_label_name]
            else:
                new_box3d_label_name = box3d_data_model.box3d_label_name

            # Map the new label name to the new label index,
            # if the new label name is not in the target class names, map to the ignore label index
            if new_box3d_label_name in self.label_index_remapper:
                new_box3d_label_index = self.label_index_remapper[new_box3d_label_name]
            else:
                new_box3d_label_index = self.ignore_label_index

            new_box3d = box3d_data_model.create_new_data_model(
                box3d_label_name=new_box3d_label_name,
                box3d_label_index=new_box3d_label_index,
            )
            new_boxes3d_data_model.append(new_box3d)

        return new_boxes3d_data_model
