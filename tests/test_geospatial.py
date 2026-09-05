"""
Unit tests for Geospatial Event Generation and Geodesic Area (FR-21 to FR-25).
"""
from datetime import datetime
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from src.geospatial.area_calculator import calculate_geodesic_polygon_area, calculate_geometry_area
from src.geospatial.geospatializer import SpillGeospatializer
from src.geospatial.event_builder import build_geojson_feature, build_geojson_feature_collection
from src.postprocessing.mask_processor import SpillComponent
from src.preprocessing.georeference import GeoreferenceTransform


def test_geodesic_area_calculation():
    # 0.1 degree x 0.1 degree box in Gulf of Mexico (~28 N, -90 W)
    # 1 deg lat ~ 111 km, 1 deg lon at 28N ~ 98 km -> 0.1 deg x 0.1 deg ~ 11.1 km * 9.8 km ~ 108 km²
    poly = box(-90.1, 28.0, -90.0, 28.1)
    area_km2, area_m2 = calculate_geodesic_polygon_area(poly)

    assert 90.0 < area_km2 < 125.0, f"Unexpected geodesic area: {area_km2} km²"
    assert np.isclose(area_m2, area_km2 * 1_000_000.0)


def test_geospatial_event_builder():
    geo_transform = GeoreferenceTransform(bounds=[-91.0, 27.0, -90.0, 28.0], width=1000, height=1000)
    geospatializer = SpillGeospatializer()

    # Create dummy component contour in pixel space
    contour = np.array([[100, 100], [150, 100], [150, 150], [100, 150]])
    comp = SpillComponent(
        component_id=1,
        mask=np.zeros((1000, 1000), dtype=bool),
        pixel_area=2500,
        mean_confidence=0.88,
        peak_confidence=0.96,
        centroid_pixel=(125.0, 125.0),
        bbox_pixel=(100, 100, 150, 150),
        contours=[contour],
    )

    event = geospatializer.create_event_from_component(
        component=comp,
        geo_transform=geo_transform,
        scene_id="S1A_GEO_TEST",
        acquisition_time=datetime(2020, 5, 12, 14, 30),
    )

    assert event is not None
    assert event.event_id.startswith("OSE-20200512-")
    assert event.area_km2 > 0.0
    assert event.confidence == 0.88
    assert event.bounding_box[0] < event.bounding_box[2]  # min_lon < max_lon
    assert event.bounding_box[1] < event.bounding_box[3]  # min_lat < max_lat

    fc = build_geojson_feature_collection([event])
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    assert fc["features"][0]["id"] == event.event_id
