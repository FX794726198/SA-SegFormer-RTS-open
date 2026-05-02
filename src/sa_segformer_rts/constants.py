"""Project defaults shared by scripts and library code."""

DEFAULT_FEATURE_NAMES = ["DEM", "EVI", "FTI", "LST", "NBR", "NDMI", "NDVI", "TCB", "TCG", "TCW"]
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_THRESHOLD = 0.4
DEFAULT_SEED = 20250917
DEFAULT_SPLIT_COUNTS = {"train": 837, "val": 179, "test": 179}

OPTICAL_DIR_CANDIDATES = [
    "optical",
    "光学遥感",
    "光学",
    "img",
    "images",
]

IMAGE_DIR_CANDIDATES = ["images", "imgaes", "IMAGES", "Images", "img", "IMG"]
LABEL_DIR_CANDIDATES = ["labels", "label", "LABELS", "Label"]

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
RASTER_EXTENSIONS = (".tif", ".tiff")
