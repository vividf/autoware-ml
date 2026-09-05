from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Float32, Int64
import numpy as np
import numpy.typing as npt
import torch


class Segmentation3DGTSample(NamedTuple):
    """Named tuple to represent a single sample of 3D segmentation GT data.

    Attributes:
      gt_semantic_mask: Point-wise labels aligned with the sample's points, shape (N, 1).
      origin_semantic_mask: Point-wise labels before any grid quantization, shape (M, 1).
        Metrics are reported at this level; ``None`` until a quantizing transform
        records what it dropped.
      origin_coord: Coordinates of those original points, shape (M, 3). Metrics bucket
        by range, so they need the pre-quantization geometry.
      inverse: For each original point, the index of the kept point representing it,
        shape (M,). Predictions are scattered back through it before scoring.
    """

    gt_semantic_mask: npt.NDArray[np.int32]  # (N, 1)
    origin_semantic_mask: npt.NDArray[np.int32] | None = None
    origin_coord: npt.NDArray[np.float32] | None = None
    inverse: npt.NDArray[np.int64] | None = None


class Segmentation3DGTBatch(NamedTuple):
    """A batch of 3D segmentation GT, flattened across samples.

    Point-wise targets concatenate the way the points do (see
    :class:`~autoware_ml.dataclasses.points_data.PointsData`), so the loss reads
    ``gt_semantic_mask`` against the model's per-point logits directly.

    Metrics are a different level: they score the *original* points, reached by
    scattering predictions through ``PointsData.inverse``. ``origin_*`` therefore stay
    separate rather than being folded into the training targets.

    Attributes:
      gt_semantic_mask: Labels aligned with the model's input points, shape (P,).
      origin_semantic_mask: Labels at the original point level, shape (O,).
      origin_coord: Original point coordinates, shape (O, 3).
      inverse: Index into the model's input points for each original point, shape (O,),
        already shifted so it indexes the concatenated batch rather than one sample.
    """

    gt_semantic_mask: Int64[torch.Tensor, " num_points"]
    origin_semantic_mask: Int64[torch.Tensor, " num_origin_points"] | None
    origin_coord: Float32[torch.Tensor, "num_origin_points 3"] | None
    inverse: Int64[torch.Tensor, " num_origin_points"] | None

    def to_device(self, device: torch.device) -> Segmentation3DGTBatch:
        """Move every tensor to ``device``."""
        return Segmentation3DGTBatch(
            gt_semantic_mask=self.gt_semantic_mask.to(device),
            origin_semantic_mask=(
                None if self.origin_semantic_mask is None else self.origin_semantic_mask.to(device)
            ),
            origin_coord=None if self.origin_coord is None else self.origin_coord.to(device),
            inverse=None if self.inverse is None else self.inverse.to(device),
        )

    @staticmethod
    def collate_gt_samples(
        gt_samples: Sequence[Segmentation3DGTSample],
    ) -> Segmentation3DGTBatch | None:
        """Concatenate per-sample segmentation targets into one batch.

        Args:
          gt_samples: Per-sample segmentation GT, in batch order.

        Returns:
          The collated batch, or ``None`` for an empty sequence.
        """
        if len(gt_samples) == 0:
            return None

        def _concat(arrays: list[npt.NDArray], dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(np.concatenate(arrays, axis=0)).to(dtype)

        masks = [np.asarray(sample.gt_semantic_mask).reshape(-1) for sample in gt_samples]
        gt_semantic_mask = _concat(masks, torch.int64)

        origin_masks = [sample.origin_semantic_mask for sample in gt_samples]
        origin_coords = [sample.origin_coord for sample in gt_samples]
        inverses = [sample.inverse for sample in gt_samples]
        has_origin = all(
            value is not None for value in (*origin_masks, *origin_coords, *inverses)
        )
        if not has_origin:
            return Segmentation3DGTBatch(
                gt_semantic_mask=gt_semantic_mask,
                origin_semantic_mask=None,
                origin_coord=None,
                inverse=None,
            )

        # `inverse` indexes each sample's kept points, so concatenating the samples means
        # shifting each block by where that sample's points start in the batch.
        kept_counts = torch.tensor([mask.shape[0] for mask in masks], dtype=torch.int64)
        starts = torch.cumsum(kept_counts, dim=0) - kept_counts
        shifted = [
            torch.from_numpy(np.asarray(inverse).reshape(-1)).to(torch.int64) + start
            for inverse, start in zip(inverses, starts.tolist())
        ]
        return Segmentation3DGTBatch(
            gt_semantic_mask=gt_semantic_mask,
            origin_semantic_mask=_concat(
                [np.asarray(m).reshape(-1) for m in origin_masks], torch.int64
            ),
            origin_coord=_concat(
                [np.asarray(c).reshape(-1, 3) for c in origin_coords], torch.float32
            ),
            inverse=torch.cat(shifted, dim=0),
        )
