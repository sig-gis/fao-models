import concurrent.futures

import os
import io

from google.api_core import exceptions, retry
import ee
import numpy as np
import tensorflow as tf
import requests


@retry.Retry()
def get_tiff_patch_url_file_point_least_cloudy(
    imageColl: ee.ImageCollection,
    point: ee.Geometry,
    bands: list,
    scale: int,
    patch_size: int,
    output_file: str,
):
    """
    Return ee.Image.getDownloadURL response and a filename as a tuple for a GeoTIFF. filename is passed through for concurrent.futures multiprocessing jobs. Uses points rather than polygons.
    args:
      imageColl: ee.ImageCollection
      point: ee.Geometry,
      bands: list,
      scale: int,
      patch_size: int,
      output_file: str,
    """
    # Create the URL to download the band values of the patch of pixels.
    point = ee.Geometry(point)
    region = point.buffer(scale * patch_size / 2, 1).bounds(1)
    
    # filter s2 collection on the fly per point to get least cloudy image
    image = imageColl.filterBounds(point).sort("CLOUDY_PIXEL_PERCENTAGE").first()
    
    url = image.getDownloadURL(
        {
            "region": region,
            "dimensions": [patch_size, patch_size],
            "format": "GEO_TIFF",
            "bands": bands,
        }
    )
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)
    response.raise_for_status()
    with open(output_file, "wb") as fd:
        fd.write(response.content)
    return 1

def write_geotiff_patch_from_points_least_cloudy(
    image,
    points,
    bands,
    scale,
    patch_size,
    output_directory,
    suffix=None,
    num_workers=10,
):
    """Writes patches inside boxes a GEE Image within a FeatureCollection of boxes to individual GeoTIFFs
    args:
      image: ee.Image 
      points: ee.FeatureCollection
      bands: list(str)
      scale: int
      patch_size: int
      output_directory: str
      suffix: str
      num_workers: int
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:  # max concurrent requests to high volume endpoint is 40
        # convert points FeatureCollection to ee.Geomtry's
        patch_pt_list = (
            points.toList(points.size())
            .map(lambda f: ee.Feature(f).geometry())
            .getInfo()
        )  # list of ee.Geometry's
        patch_pt_list_filenames = [
            os.path.join(output_directory, f"patch_pt{list_index}_{suffix}.tif")
            for list_index in list(range(0, points.size().getInfo()))
        ]  # list of filenames
        pt_filename = zip(patch_pt_list, patch_pt_list_filenames)
        # # don't write patches that already exist on disk
        pt_filename = [pf for pf in pt_filename if not os.path.exists(pf[1])]
        # end this stuff
        futures = {
            executor.submit(
                get_tiff_patch_url_file_point_least_cloudy,
                image,
                pt,
                bands,
                scale,
                patch_size,
                filename,
            ): (pt, filename)
            for (
                pt,
                filename,
            ) in pt_filename  # zip(patch_pt_list,patch_pt_list_filenames)
        }
        try:
            for future in concurrent.futures.as_completed(futures, timeout=120):
                try:
                    # get result of future with timeout of 120s
                    result = future.result(timeout=120)

                except concurrent.futures.TimeoutError:
                    print("The task exceeded 2-minute limit and was cancelled")
                except Exception as e:
                    print(f"Loop - Generated an exception: {e}")
        except Exception as e:
            # TODO - this still generates an exception when doing a long running (and finishing task)
            #        Test to see if removing try/except will still allow task to finish.
            print(f"Outer - Generated an exception: {e}")


@retry.Retry()
def get_tiff_patch_url_file_point(
    image: ee.Image,
    point: ee.Geometry,
    bands: list,
    scale: int,
    patch_size: int,
    output_file: str,
):
    """
    Return ee.Image.getDownloadURL response and a filename as a tuple for a GeoTIFF. filename is passed through for concurrent.futures multiprocessing jobs. Uses points rather than polygons.
    args:
      image: ee.Image
      box: ee.Geometry
      bands: list(str)
      output_file: str
    """
    # Create the URL to download the band values of the patch of pixels.
    point = ee.Geometry(point)
    region = point.buffer(scale * patch_size / 2, 1).bounds(1)

    url = image.getDownloadURL(
        {
            "region": region,
            "dimensions": [patch_size, patch_size],
            "format": "GEO_TIFF",
            "bands": bands,
        }
    )
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)
    response.raise_for_status()
    with open(output_file, "wb") as fd:
        fd.write(response.content)
    return 1


def write_geotiff_patch_from_points(
    image,
    points,
    bands,
    scale,
    patch_size,
    output_directory,
    suffix=None,
    num_workers=10,
):
    """Writes patches inside boxes a GEE Image within a FeatureCollection of boxes to individual GeoTIFFs
    args:
      image: ee.Image 
      points: ee.FeatureCollection
      bands: list(str)
      scale: int
      patch_size: int
      output_directory: str
      suffix: str
      num_workers: int
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:  # max concurrent requests to high volume endpoint is 40
        # convert points FeatureCollection to ee.Geomtry's
        patch_pt_list = (
            points.toList(points.size())
            .map(lambda f: ee.Feature(f).geometry())
            .getInfo()
        )  # list of ee.Geometry's
        patch_pt_list_filenames = [
            os.path.join(output_directory, f"patch_pt{list_index}_{suffix}.tif")
            for list_index in list(range(0, points.size().getInfo()))
        ]  # list of filenames
        pt_filename = zip(patch_pt_list, patch_pt_list_filenames)
        # # don't write patches that already exist on disk
        pt_filename = [pf for pf in pt_filename if not os.path.exists(pf[1])]
        # end this stuff
        futures = {
            executor.submit(
                get_tiff_patch_url_file_point,
                image,
                pt,
                bands,
                scale,
                patch_size,
                filename,
            ): (pt, filename)
            for (
                pt,
                filename,
            ) in pt_filename  # zip(patch_pt_list,patch_pt_list_filenames)
        }
        try:
            for future in concurrent.futures.as_completed(futures, timeout=120):
                try:
                    # get result of future with timeout of 120s
                    result = future.result(timeout=120)

                except concurrent.futures.TimeoutError:
                    print("The task exceeded 2-minute limit and was cancelled")
                except Exception as e:
                    print(f"Loop - Generated an exception: {e}")
        except Exception as e:
            # TODO - this still generates an exception when doing a long running (and finishing task)
            #        Test to see if removing try/except will still allow task to finish.
            print(f"Outer - Generated an exception: {e}")

def write_tfrecord_batch(image, patch_size, points, scale, output_file):
    """Writes patches at a set of points to a TFRecord file, using ee.data.ComputePixels
    args:
      image: ee.Image
      patch_size: int
      points: python list of ee.Geometry.Point objects, easily done with `pointFC.aggregate_array('.geo').getInfo()`
      scale: int
      output_file: str
    returns: None
    """
    # REPLACE WITH YOUR BUCKET!
    OUTPUT_FILE = output_file

    # Output resolution in meters.
    SCALE = scale

    # Pre-compute a geographic coordinate system.
    proj = ee.Projection("EPSG:4326").atScale(SCALE).getInfo()

    # Get scales in degrees out of the transform.
    SCALE_X = proj["transform"][0]
    SCALE_Y = -proj["transform"][4]

    # Patch size in pixels.
    PATCH_SIZE = patch_size

    # Offset to the upper left corner.
    OFFSET_X = -SCALE_X * PATCH_SIZE / 2
    OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2

    # Request template for ee.data.ComputePixels
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

    # Blue, green, red, NIR, AOT.
    FEATURES = (
        image.bandNames().getInfo()
    )  # ['B2_median', 'B3_median', 'B4_median', 'B8_median', 'AOT_median']

    # Specify the size and shape of patches expected by the model.
    KERNEL_SHAPE = [PATCH_SIZE, PATCH_SIZE]
    COLUMNS = [
        tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
    ]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    EXECUTOR = concurrent.futures.ThreadPoolExecutor(
        max_workers=40
    )  # max concurrent requests to high volume endpoint

    # functions for batch .tfrecord writer workflow
    @retry.Retry()
    def get_patch(coords, image, format="NPY"):
        """Uses ee.data.ComputePixels() to get a patch centered on the coordinates, as a numpy array."""
        request = dict(REQUEST)
        request["fileFormat"] = format
        request["expression"] = image
        request["grid"]["affineTransform"]["translateX"] = coords[0] + OFFSET_X
        request["grid"]["affineTransform"]["translateY"] = coords[1] + OFFSET_Y
        return np.load(io.BytesIO(ee.data.computePixels(request)))

    def get_sample_coords(roi, n):
        """ "Get a random sample of N points in the ROI."""
        points = ee.FeatureCollection.randomPoints(region=roi, points=n, maxError=1)
        return points.aggregate_array(".geo").getInfo()

    def array_to_example(structured_array):
        """ "Serialize a structured numpy array into a tf.Example proto."""
        feature = {}
        for f in FEATURES:
            feature[f] = tf.train.Feature(
                float_list=tf.train.FloatList(value=structured_array[f].flatten())
            )
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tf_dataset(image, sample_points, file_name):
        """ "Write patches at the sample points into one TFRecord file."""
        future_to_point = {
            EXECUTOR.submit(get_patch, point["coordinates"], image): point
            for point in sample_points
        }

        # Optionally compress files.
        writer = tf.io.TFRecordWriter(file_name)

        for future in concurrent.futures.as_completed(future_to_point):
            point = future_to_point[future]
            try:
                np_array = future.result()
                example_proto = array_to_example(np_array)
                writer.write(example_proto.SerializeToString())
                writer.flush()
            except Exception as e:
                # print(e)
                pass

        writer.close()

    # write patches to .tfrecord file
    write_tf_dataset(image, points, OUTPUT_FILE)
    return
