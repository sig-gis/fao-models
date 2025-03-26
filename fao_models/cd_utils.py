import geopandas as gpd
import os
import ee
import io
from io import StringIO
from google.api_core import retry
import numpy as np
from shapely import Point
import shapely
import rasterio
from rasterio.transform import from_bounds


classes ={
    'Stable Non Forest': 0,
    'Stable Forest': 1,
    'Forest Loss': 2,
    'Forest Gain': 3,
}


# read feature geoJSON (these are per GEZ)
def transform(gdf,t1start,t1end,t2start,t2end):
    list_of_features = []
    for i,row in gdf.iterrows():
        f_dict = {
            'PLOTID': row.PLOTID,
            'geometry': row.geometry,
            'top-left': Point(
                [list(row.geometry.bounds)[0],
                 list(row.geometry.bounds)[3]]),
            'sample_dates': 
            {
            't1start':t1start,
            't1end':t1end, 
            't2start': t2start,
            't2end': t2end,
            }
            }
        
        list_of_features.append(f_dict)
    
    return list_of_features

def get_landsat_composite(region:shapely.Polygon,
                          start:str,
                          end:str):
    # Define the region of interest as a bounding box.
    region = ee.Geometry.Polygon(list(region.exterior.coords))
    band_mapper = {
        'l5':{
            'bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            'band_names': ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']},
        'l7':{
            'bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            'band_names': ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']},
        'l8':{
            'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'band_names': ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']},
        'l9':{
            'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'band_names': ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        }
    }
    
    # Applies scaling factors.
    def apply_scale_factorsl5l7(image):
       optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
       thermal_bands = image.select('ST_B6').multiply(0.00341802).add(149.0)
       return image.addBands(optical_bands, None, True).addBands(
           thermal_bands, None, True
           )
    def apply_scale_factorsl8l9(image):
       optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
       thermal_bands = image.select('ST_B10').multiply(0.00341802).add(149.0)
       return image.addBands(optical_bands, None, True).addBands(
           thermal_bands, None, True
           )
    
    def qal5l7(image):
        """Custom QA masking method for Landsat9 surface reflectance dataset"""
        qa_band = image.select("QA_PIXEL")
        qa_flag = int('111111',2)
        sat_mask = image.select('QA_RADSAT').eq(0);
        mask = qa_band.bitwiseAnd(qa_flag).eq(0).And(sat_mask)
        return apply_scale_factorsl5l7(image).updateMask(mask)
    
    def qal8l9(image):
        """Custom QA masking method for Landsat9 surface reflectance dataset"""
        qa_band = image.select("QA_PIXEL")
        qa_flag = int('111111',2)
        sat_mask = image.select('QA_RADSAT').eq(0);
        mask = qa_band.bitwiseAnd(qa_flag).eq(0).And(sat_mask)
        return apply_scale_factorsl8l9(image).updateMask(mask)
    
    # Create a Landsat image collection for the specified date range.
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(region)
          .filterDate(start, end)
          .map(qal5l7)
          .select(band_mapper['l5']['bands'], band_mapper['l5']['band_names'])
          )
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start, end)
            .map(qal5l7)
            .select(band_mapper['l7']['bands'], band_mapper['l7']['band_names'])
            )
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start, end)
            .map(qal8l9)
            .select(band_mapper['l8']['bands'], band_mapper['l8']['band_names'])
            )
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start, end)
            .map(qal8l9)
            .select(band_mapper['l9']['bands'], band_mapper['l9']['band_names'])
            )
    
    collection = l5.merge(l7).merge(l8).merge(l9)
    image = collection.median()
    return image

def get_arr_from_geom_centr(image:ee.Image,
                 geom:shapely.Geometry,
                 size:int,
                 gsd:int,):
    
    def get_innermost_H_W(array, height=32, width=32):
        # Get the shape of the input array
        rows, cols = array.shape
        
        # Calculate the starting indices
        start_row = (rows - height) // 2
        start_col = (cols - width) // 2
        
        # Slice the array to get the innermost 32x32 values
        inner_array = array[start_row:start_row + height, start_col:start_col + width]
        
        return inner_array

    PATCH_SIZE = size
    SCALE = gsd
    proj = ee.Projection("EPSG:4326").atScale(SCALE).getInfo()
    SCALE_X = proj["transform"][0]
    SCALE_Y = -proj["transform"][4]

    # Offset to the upper left corner.
    OFFSET_X = -SCALE_X * PATCH_SIZE / 2
    OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2

    translateX = list(geom.bounds)[0] + OFFSET_X
    translateY = list(geom.bounds)[3] + OFFSET_Y

    



    # request image as numpy array, do some reshaping to 32,32
    request = {
            'expression': image,
            'fileFormat': 'NUMPY_NDARRAY',
            "grid": {
            "dimensions": {"width": PATCH_SIZE, "height": PATCH_SIZE},
            "affineTransform": {
                "scaleX": SCALE_X,
                "shearX": 0,
                "shearY": 0,
                "scaleY": SCALE_Y,
                "translateX": translateX,
                "translateY": translateY
            },
            "crsCode": proj["crs"],
        }
        }
    data = ee.data.computePixels(request)
    # print(data.shape)
    # Assuming data is a structured array with multiple bands
    bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    data_innermost = {band: get_innermost_H_W(data[band],size,size) for band in bands}
    img = np.stack([data_innermost[band] for band in bands])
    
    
    
    return img


def parse_shp_to_latlon(file,id_field:str='PLOTID'):
    gdf = gpd.read_file(file)
    gdf.geometry.to_crs(epsg=3857)
    gdf.loc[:,'centroid'] = gdf.geometry.centroid
    gdf.loc[:,'lonlat'] = gdf.centroid.apply(lambda x: [x.x, x.y])
    return gdf[[id_field, 'lonlat']].values.tolist()
        
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

