from __future__ import annotations

import io
import ee
from google.api_core import exceptions, retry
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests

SCALE = 10  # meters per pixel


def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )


def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    CLOUD_BIT = 10
    CIRRUS_CLOUD_BIT = 11
    bit_mask = (1 << CLOUD_BIT) | (1 << CIRRUS_CLOUD_BIT)
    mask = image.select("QA60").bitwiseAnd(bit_mask).eq(0)
    return image.updateMask(mask)


def get_input_image(year: int, default_value: float = 1000.0) -> ee.Image:
    landsat = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(f"{year}-1-1", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_sentinel2_clouds)
        .select("B4", "B3", "B2")
        .median()
        .unmask(default_value)
    )
    # modis = ee.ImageCollection("MODIS/061/MCD43A4").filterDate(f"{year}-1-1", f"{year}-12-31").select("Nadir_Reflectance_B.*").median()
    firms_collection = (
        ee.ImageCollection("FIRMS")
        .filterDate(f"{year}-1-1", f"{year}-12-31")
        .select("T21")
        .median()
    )
    env_collection = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(f"{year}-1-1", f"{year}-12-31")
        .select(["tmmx", "tmmn", "pr", "vs", "sph"])
        .median()
    )
    veg_collection = (
        ee.ImageCollection("MODIS/006/MOD13A2")
        .filterDate(f"{year}-1-1", f"{year}-12-31")
        .select("NDVI")
        .median()
    )
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    merged_image = (
        landsat.addBands(firms_collection)
        .addBands(env_collection)
        .addBands(veg_collection)
        .addBands(elevation)
    )
    return merged_image.unmask(0).float()


def get_label_image() -> ee.Image:
    label_asset_id = "projects/wildfire-risk-prediction/assets/whp_2020"
    # Load the image from the asset
    return ee.Image(label_asset_id)


def get_input_patch(year: int, lonlat, patch_size: int) -> np.ndarray:
    image = get_input_image(year)
    patch = get_patch(image, lonlat, patch_size, SCALE)
    return structured_to_unstructured(patch)


def get_label_patch(lonlat, patch_size: int) -> np.ndarray:
    image = get_label_image()
    patch = get_patch(image, lonlat, patch_size, SCALE)
    return structured_to_unstructured(patch)


@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(image: ee.Image, lonlat, patch_size: int, scale: int) -> np.ndarray:
    point = ee.Geometry.Point(lonlat)
    url = image.getDownloadURL(
        {
            "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        }
    )
    # If we get "429: Too Many Requests" errors, it's safe to retry the request.
    # The Retry library only works with `google.api_core` exceptions.
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    # Still raise any other exceptions to make sure we got valid data.
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True)

