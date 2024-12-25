import tifffile as tiff
import numpy as np

def split_tiff(image_path, tile_size):
    with tiff.TiffFile(image_path) as tif:
        image = tif.asarray()
        height, width = image.shape[:2]
        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                tile = image[j:j + tile_size, i:i + tile_size]
                tiff.imwrite(f"tile_{i}_{j}.tiff", tile)

split_tiff("2023 강천보 갈수기 분석_231120_transparent_mosaic_group1.tif", 256)