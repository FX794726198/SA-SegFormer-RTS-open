"""SA-SegFormer tooling for retrogressive thaw slump segmentation.

The package initializer is intentionally lightweight so dataset manifest tools
can run before heavy training dependencies such as segmentation-models-pytorch
or albumentations are installed.
"""

from .constants import DEFAULT_FEATURE_NAMES, DEFAULT_IMAGE_SIZE, DEFAULT_SPLIT_COUNTS, DEFAULT_THRESHOLD

__all__ = ["DEFAULT_FEATURE_NAMES", "DEFAULT_IMAGE_SIZE", "DEFAULT_SPLIT_COUNTS", "DEFAULT_THRESHOLD"]
