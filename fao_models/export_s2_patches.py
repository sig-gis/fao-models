import ee
import os
import google.auth
from serving import *
import os
import argparse

# os.environ['TF_ENABLE_ONEDNN_OPTS=0']

PROJECT = "pc530-fao-fra-rss"  # change to your cloud project name
# ee.Initialize(project=PROJECT)

## INIT WITH HIGH VOLUME ENDPOINT
credentials, _ = google.auth.default()
ee.Initialize(
    credentials,
    project=PROJECT,
    opt_url="https://earthengine-highvolume.googleapis.com",
)

# USE HEXAGONS TO MAKE PATCH BOUNDS ######################################################
all_hex = ee.FeatureCollection(
    "projects/pc530-fao-fra-rss/assets/reference/hexWCenPropertiesTropics"
)
# print(all_hex.size().getInfo())
# print(all_hex.limit(1).getInfo()['features'][0]['properties'])

hexForest = all_hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 1), ee.Filter.eq("LU18CEN", "Forest"))
)
# print('pureForest size',hexForest.size().getInfo())
# print('LU18CEN values pure forest',hexForest.aggregate_histogram('LU18CEN').getInfo())

hexNonForest = all_hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 0), ee.Filter.neq("LU18CEN", "Forest"))
)
# print('pureNonForest size',hexNonForest.size().getInfo())
# print('LU18CEN values pure nonForest',hexNonForest.aggregate_histogram('LU18CEN').getInfo())

# hexForest = hexForest.randomColumn().limit(10000,'random',False)
# hexNonForest = hexNonForest.randomColumn().limit(10000,'random',False)
FNFhex = hexForest.merge(hexNonForest)
sample_size_total = FNFhex.size().getInfo()
# print(sample_size_total)


# create 320 x 320 m box for image patches (32x32 px patches for training)
def hex_patch_box(fc, size):
    def per_hex(f):
        centroid = f.geometry().centroid()
        patch_box = centroid.buffer(size / 2).bounds()
        return ee.Feature(patch_box)

    return fc.map(per_hex)


patch_boxes = hex_patch_box(FNFhex, 320)

# Finally, for actual workflow at botom we only need the centroids of each hexagon to generate the image patchess
FNFhex_centroids = FNFhex.map(lambda h: ee.Feature(h.geometry().centroid()))
# print(FNFhex_centroids.first().getInfo())

# image patch generation from hexagon centroid
hexLabel = (
    ee.Image(0).paint(hexForest, 1).paint(hexNonForest, 2).selfMask().rename("class")
)

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
    s2.filterDate("2017-01-01", "2019-12-31")
    .linkCollection(csPlus, [QA_BAND])
    .map(lambda img: img.updateMask(hexLabel))
    .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
    .median()
    .addBands(hexLabel)
    .select(["B4", "B3", "B2", "B8", "class"], ["R", "G", "B", "N", "class"])
)  # B G R classlabel


## TESTING ################################################################
def main():
    # initalize new cli parser
    parser = argparse.ArgumentParser(description="Export S2 image patches.")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="path to config file",
    )

    parser.add_argument(
        "-f",
        "--forest",
        dest="forest",
        action="store_true",
        help="export forest labeled patches",
        required=False,
    )

    parser.add_argument(
        "-nf",
        "--nonforest",
        dest="nonforest",
        action="store_true",
        help="export nonforest labeled patches",
        required=False,
    )
    args = parser.parse_args()

    parser.set_defaults(forest=False)
    parser.set_defaults(nonforest=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # we have about a 1/3 to 2/3 split of forest / nonforest makeup of total hexagons
    ee_points_forest = (
        hexForest.map(lambda h: ee.Feature(h.geometry().centroid()))
        .randomColumn()
        .sort("random")
    )
    ee_points_nonforest = (
        hexNonForest.map(lambda h: ee.Feature(h.geometry().centroid()))
        .randomColumn()
        .sort("random")
    )

    if not args.forest and not args.nonforest:
        print("Please specify --forest and/or --nonforest")
        exit()

    if args.forest:
        # for i in [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: # first chunk finishes then hangs..
        #   print(f'exporting patches from points chunk {i}')
        #   points_chunk = ee_points_forest.filter(ee.Filter.And(ee.Filter.gt('random',i),ee.Filter.lte('random',i+0.1)))
        #   # print(points_chunk.size().getInfo())
        # write_geotiff_patch_from_points_v2(sampleImage,points_chunk,['R','G','B','N'],10,32,output_directory=args.output_dir, suffix='forest')
        write_geotiff_patch_from_points_v2(
            sampleImage,
            ee_points_forest,
            ["R", "G", "B", "N"],
            10,
            32,
            output_directory=args.output_dir,
            suffix="forest",
        )

    if args.nonforest:
        write_geotiff_patch_from_points_v2(
            sampleImage,
            ee_points_nonforest,
            ["R", "G", "B", "N"],
            10,
            32,
            output_directory=args.output_dir,
            suffix="nonforest",
        )


if __name__ == "__main__":
    main()
