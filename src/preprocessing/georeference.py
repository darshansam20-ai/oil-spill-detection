"""
Georeferencing and spatial coordinate transformation utilities (PRD FR-13, FR-21).
Converts raster pixel indices (row, col) to Geographic (longitude, latitude) coordinates (EPSG:4326).
"""
from typing import List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import Affine, xy, rowcol
from rasterio.warp import transform as warp_transform
from src.utils.logger import get_logger

logger = get_logger("preprocessing.georeference")


class GeoreferenceTransform:
    """Encapsulates affine transformation and CRS projection for a SAR raster scene."""

    def __init__(
        self,
        transform: Optional[Affine] = None,
        crs: str = "EPSG:4326",
        bounds: Optional[List[float]] = None,
        width: int = 1024,
        height: int = 1024,
        is_fallback: bool = False,
    ):
        self.crs = crs or "EPSG:4326"
        self.width = width
        self.height = height
        self.is_fallback = is_fallback

        if transform is not None:
            self.transform = transform
        elif bounds is not None and len(bounds) == 4:
            min_x, min_y, max_x, max_y = bounds
            x_res = (max_x - min_x) / width
            y_res = (max_y - min_y) / height
            # North-up affine transform: [x_res, 0, min_x, 0, -y_res, max_y]
            self.transform = Affine(x_res, 0.0, min_x, 0.0, -y_res, max_y)
        else:
            self.is_fallback = True
            # Default fallback identity transform around Gulf of Mexico
            self.transform = Affine(0.0001, 0.0, -90.0, 0.0, -0.0001, 28.0)

    @classmethod
    def from_geotiff(cls, file_path: str) -> "GeoreferenceTransform":
        """Read transform and CRS directly from a GeoTIFF raster file."""
        with rasterio.open(file_path) as src:
            is_valid_crs = bool(src.crs and src.transform and not src.transform.is_identity)
            return cls(
                transform=src.transform,
                crs=str(src.crs) if src.crs else "EPSG:4326",
                width=src.width,
                height=src.height,
                is_fallback=not is_valid_crs,
            )

    def pixel_to_geo(self, row: float, col: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates (row, col) to geographic (longitude, latitude) in WGS84 (EPSG:4326).
        """
        x, y = xy(self.transform, row, col, offset="center")
        
        # If CRS is projected (e.g. UTM / EPSG:326xx), warp to WGS84 EPSG:4326
        if self.crs and self.crs.upper() not in ["EPSG:4326", "WGS 84", "OGC:CRS84", "+PROJ=LATLONG"]:
            try:
                xs, ys = warp_transform(self.crs, "EPSG:4326", [x], [y])
                return float(xs[0]), float(ys[0])
            except Exception as e:
                logger.warning(f"Failed to reproject from {self.crs} to EPSG:4326: {e}")
        
        return float(x), float(y)

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert geographic (longitude, latitude) to pixel coordinates (row, col).
        """
        x, y = lon, lat
        if self.crs and self.crs.upper() not in ["EPSG:4326", "WGS 84", "OGC:CRS84"]:
            try:
                xs, ys = warp_transform("EPSG:4326", self.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            except Exception as e:
                logger.warning(f"Failed to warp from EPSG:4326 to {self.crs}: {e}")
        
        r, c = rowcol(self.transform, x, y)
        return int(r), int(c)

    def get_bounds(self) -> List[float]:
        """
        Get geographic bounding box in WGS84: [min_lon, min_lat, max_lon, max_lat].
        """
        lon1, lat1 = self.pixel_to_geo(0, 0)
        lon2, lat2 = self.pixel_to_geo(self.height, self.width)
        return [
            min(lon1, lon2),
            min(lat1, lat2),
            max(lon1, lon2),
            max(lat1, lat2),
        ]
