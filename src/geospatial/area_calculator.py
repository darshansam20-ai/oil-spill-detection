"""
Geodesic Area Calculation Module (PRD FR-23).
Computes exact physical surface area of geographic polygons using ellipsoidal WGS84 geodesic geometry.
Avoids naive degree-to-meter distortions.
"""
from typing import Tuple
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Geod

# Standard WGS84 ellipsoid model
WGS84_GEOD = Geod(ellps="WGS84")


def calculate_geodesic_polygon_area(polygon: Polygon) -> Tuple[float, float]:
    """
    Calculate geodesic surface area of a Shapely Polygon in square kilometers (km²) and square meters (m²).
    
    Args:
        polygon: Shapely Polygon in geographic (longitude, latitude) coordinates.
        
    Returns:
        Tuple of (area_km2, area_m2).
    """
    if polygon.is_empty or not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return 0.0, 0.0

    # Exterior ring geodesic area
    ext_lons, ext_lats = polygon.exterior.xy
    area_m2, _ = WGS84_GEOD.polygon_area_perimeter(ext_lons, ext_lats)
    area_m2 = abs(area_m2)

    # Subtract interior holes
    for interior in polygon.interiors:
        int_lons, int_lats = interior.xy
        hole_m2, _ = WGS84_GEOD.polygon_area_perimeter(int_lons, int_lats)
        area_m2 -= abs(hole_m2)

    area_m2 = max(0.0, area_m2)
    area_km2 = area_m2 / 1_000_000.0
    return float(area_km2), float(area_m2)


def calculate_geometry_area(geom: MultiPolygon | Polygon) -> Tuple[float, float]:
    """
    Calculate total geodesic area for Polygon or MultiPolygon.
    """
    if isinstance(geom, Polygon):
        return calculate_geodesic_polygon_area(geom)
    elif isinstance(geom, MultiPolygon):
        tot_km2 = 0.0
        tot_m2 = 0.0
        for poly in geom.geoms:
            km2, m2 = calculate_geodesic_polygon_area(poly)
            tot_km2 += km2
            tot_m2 += m2
        return tot_km2, tot_m2
    else:
        return 0.0, 0.0
