import ee
import os
import google.auth
from serving import write_geotiff_patch_from_boxes, write_tfrecord_batch, write_geotiff_patch_from_points
# os.environ['TF_ENABLE_ONEDNN_OPTS=0']

PROJECT = 'pc530-fao-fra-rss' # change to your cloud project name
ee.Initialize(project=PROJECT)

## INIT WITH HIGH VOLUME ENDPOINT
credentials, _ = google.auth.default()
ee.Initialize(credentials, project=PROJECT, opt_url='https://earthengine-highvolume.googleapis.com')

# USE HEXAGONS TO MAKE PATCH BOUNDS ######################################################
all_hex = ee.FeatureCollection("projects/pc530-fao-fra-rss/assets/reference/hexWCenPropertiesTropics")
# print(all_hex.size().getInfo())
# print(all_hex.limit(1).getInfo()['features'][0]['properties'])

hexForest = all_hex.filter(ee.Filter.And(ee.Filter.eq('FOREST',1),ee.Filter.eq('LU18CEN','Forest')))
# print('pureForest size',hexForest.size().getInfo())
# print('LU18CEN values pure forest',hexForest.aggregate_histogram('LU18CEN').getInfo())

hexNonForest = all_hex.filter(ee.Filter.And(ee.Filter.eq('FOREST',0),ee.Filter.neq('LU18CEN','Forest')))
# print('pureNonForest size',hexNonForest.size().getInfo())
# print('LU18CEN values pure nonForest',hexNonForest.aggregate_histogram('LU18CEN').getInfo())

# hexForest = hexForest.randomColumn().limit(10000,'random',False)
# hexNonForest = hexNonForest.randomColumn().limit(10000,'random',False)
FNFhex = hexForest.merge(hexNonForest)
sample_size_total = FNFhex.size().getInfo()
# print(sample_size_total)

# create 320 x 320 m box for image patches (32x32 px patches for training)
def hex_patch_box(fc,size):
  def per_hex(f):
    centroid = f.geometry().centroid()
    patch_box = centroid.buffer(size/2).bounds()
    return ee.Feature(patch_box)
  return fc.map(per_hex)

patch_boxes = hex_patch_box(FNFhex,320)

# Finally, for actual workflow at botom we only need the centroids of each hexagon to generate the image patchess
FNFhex_centroids = FNFhex.map(lambda h: ee.Feature(h.geometry().centroid()))
# print(FNFhex_centroids.first().getInfo())

# image patch generation from hexagon centroid
hexLabel = ee.Image(0).paint(hexForest,1).paint(hexNonForest,2).selfMask().rename('class')

## MAKE S2 COMPOSITE IN HEXAGONS ##########################################
# Using Cloud Score + for cloud/cloud-shadow masking
# Harmonized Sentinel-2 Level 2A collection.
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
# Level 1C data and can be applied to either L1C or L2A collections.
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

# Use 'cs' or 'cs_cdf', depending on your use case; see docs for guidance.
QA_BAND = 'cs_cdf'

# The threshold for masking; values between 0.50 and 0.65 generally work well.
# Higher values will remove thin clouds, haze & cirrus shadows.
CLEAR_THRESHOLD = 0.50;

# Make a clear median composite.
sampleImage = (s2
    .filterDate('2017-01-01', '2019-12-31')
    .linkCollection(csPlus, [QA_BAND])
    .map(lambda img: img.updateMask(hexLabel))
    .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
    .median()
    .addBands(hexLabel)
    .select(['B4','B3','B2','B8','class'],['R','G','B','N','class'])) # B G R classlabel

## TESTING ##
 #TODO: save files specifically in data/ directory
# Get the current working directory
cwd = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(cwd)
# Set the data directory path
data_dir = os.path.join(parent_dir, 'data')

# test_boxes = patch_boxes.limit(10)
# write_geotiff_patch_from_boxes(sampleImage,test_boxes,['R','G','B','N','class'])

# test_points = FNFhex_centroids.limit(10).aggregate_array('.geo').getInfo()
# write_tfrecord_batch(sampleImage, 32, test_points, 10, 'test_tfrecord_batch')

test_ee_points = FNFhex_centroids.limit(20000)
write_geotiff_patch_from_points(sampleImage,test_ee_points,['R','G','B','N','class'],10,32,output_directory=data_dir)