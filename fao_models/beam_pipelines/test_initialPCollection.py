#%%
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import unittest
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import logging
import geopandas
logger = logging.getLogger(__name__)

# load gdf and compute centroid from geometry
gdf = geopandas.read_file('C:\\Users\\kyle\\Downloads\\ALL_centroids_completed_v1_\\ALL_centroids_completed_v1_.shp')
gdf.loc[:,'centroid'] = gdf.geometry.centroid
print(gdf.head())
#%%
# convert centroid (a GeoSeries geometry), to a native python list of lat,lon)
gdf.loc[:,'latlon'] = gdf.centroid.apply(lambda x: [x.y, x.x])
print(gdf.dtypes)
print(gdf.head())
#%%
# construct list of global_id, latlon tuples for the pipeline
features = gdf[['global_id', 'latlon']].values.tolist()
print(features[:5])


#%%
# https://beam.apache.org/documentation/pipelines/test-your-pipeline/#testing-transforms
expected_output = features[:5]
def test_pipe(argv=None, save_main_session=True):
  """Main entry point;"""
  # read in a gdf and construct begnning PCollection from gdf in-memory  

  # do we need to convert each record in gdf to a list or dict? 
  with TestPipeline(runner=beam.runners.DirectRunner()) as p:
    
      pipe_features = p | beam.Create(features[:5]) # if you change this to features[:6] the test will raise AssertionError
      assert_that(pipe_features,equal_to(expected_output), label='check features')
    

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  test_pipe()
