import geopandas as gpd
import ee
import io
from google.api_core import retry
import numpy as np

def parse_shp_to_latlon(file,id_field:str='PLOTID'):
    gdf = gpd.read_file(file)
    gdf.geometry.to_crs(epsg=3857)
    gdf.loc[:,'centroid'] = gdf.geometry.centroid
    gdf.loc[:,'lonlat'] = gdf.centroid.apply(lambda x: [x.x, x.y])
    return gdf[[id_field, 'lonlat']].values.tolist()

def get_ee_img(coords):
    """retrieve s2 image composite from ee at given coordinates. coords is a tuple of (lon, lat) in degrees."""
    ## MAKE S2 COMPOSITE IN HEXAGONS ##########################################
    # Using Cloud Score + for cloud/cloud-shadow masking
    # Harmonized Sentinel-2 Level 2A collection.
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
    # Level 1C data and can be applied to either L1C or L2A collections.
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    # Use 'cs' or 'cs_cdf', depending on your use case; see docs for guidance.
    QA_BAND = "cs_cdf"

    # The threshold for masking; values between 0.50 and 0.65 generally work well.
    # Higher values will remove thin clouds, haze & cirrus shadows.
    CLEAR_THRESHOLD = 0.50

    # Make a clear median composite.
    sampleImage = (
        s2.filterDate("2023-01-01", "2023-12-31")
        .filterBounds(ee.Geometry.Point(coords[0], coords[1]).buffer(64*10)) # only images touching 64 pixel centroid buffer
        .linkCollection(csPlus, [QA_BAND])
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
        .median()
        .select(["B4", "B3", "B2", "B8"], ["R", "G", "B", "N"])
    )
    return sampleImage
        
@retry.Retry()
def get_patch_numpy(coords, image, format="NPY"):
    """Uses ee.data.ComputePixels() to get a 32x32 patch centered on the coordinates, as a numpy array."""

    # Output resolution in meters.
    SCALE = 10

    # Pre-compute a geographic coordinate system.
    proj = ee.Projection("EPSG:4326").atScale(SCALE).getInfo()

    # Get scales in degrees out of the transform.
    SCALE_X = proj["transform"][0]
    SCALE_Y = -proj["transform"][4]

    # Patch size in pixels.
    PATCH_SIZE = 32

    # Offset to the upper left corner.
    OFFSET_X = -SCALE_X * PATCH_SIZE / 2
    OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2

    REQUEST = {
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": PATCH_SIZE, "height": PATCH_SIZE},
            "affineTransform": {
                "scaleX": SCALE_X,
                "shearX": 0,
                "shearY": 0,
                "scaleY": SCALE_Y,
            },
            "crsCode": proj["crs"],
        },
    }

    request = dict(REQUEST)
    request["fileFormat"] = format
    request["expression"] = image
    request["grid"]["affineTransform"]["translateX"] = coords[0] + OFFSET_X
    request["grid"]["affineTransform"]["translateY"] = coords[1] + OFFSET_Y
    return np.load(io.BytesIO(ee.data.computePixels(request)))

def to_tensor(patch):
    """
    Converts a numpy array to a tf tensor
    """
    from numpy.lib.recfunctions import structured_to_unstructured

    unstruct = structured_to_unstructured(patch) # converts to CHW shape
    rescaled = unstruct.astype(np.float64) / 10000 # scale it 
    reshaped = np.reshape(rescaled, (1, 32, 32, 4)) # batch it
    return reshaped

def make_inference(model,tensor):
    """returns model prediction on the provided tensor"""
    prob = round(float(model(tensor).numpy()),2)
    prediction = 1 if prob > 0.5 else 0
    return [prob, prediction]

