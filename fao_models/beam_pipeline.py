import os
from pathlib import Path
import logging
import argparse
import datetime

from types import SimpleNamespace
import csv
import io

import pandas as pd
import geopandas as gpd

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import WriteToText

from beam_utils import parse_shp_to_latlon
from common import load_yml


# pipeline to have these general steps

# 1. Read in data from SHP (hexagons were provided as SHP and CSV but CSV has no geom column, centroids only came as SHP file)
# 2. parse data into row-wise elements of (global id, [lon,lat]) - rest of pipeline passes these elements through
# 3. download imagery for each element and convert to a tensor
# 4. load model and run inference on tensor to return prediction value
# 5. write prediction value to new CSV file with (global id, lat, long, prediction value)
# 6. join CSV file(s) with model predictions back to the original SHP File and export as a new SHP file

logging.basicConfig(
    filename=f"forest-classifier-beam-{datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}.log",
    encoding="utf-8",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)

class GetPatch(beam.DoFn):
    def __init__(self):
        super().__init__()

    def setup(self):
        import ee
        import google.auth

        credentials, _ = google.auth.default()
        ee.Initialize(
            credentials,
            project="pc530-fao-fra-rss",
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
        return super().setup()

    def process(self, element):
        from beam_utils import get_ee_img, get_patch_numpy, to_tensor

        # element is a tuple of (global_id, [lon,lat])
        global_id = element[0]
        coords = element[1]
        logging.info(f"Getting Imagery for {global_id} at {coords}")
        image = get_ee_img(coords)
        patch = get_patch_numpy(coords, image)
        patch_tensor = to_tensor(patch)

        yield {
            "PLOTID": global_id,
            "lon": coords[0],
            "lat": coords[1],
            "patch": patch_tensor,
        }


class Predict(beam.DoFn):
    def __init__(self, config_path):
        from common import load_yml

        # from _types import Config # Config was a dataclass subclass in Johns repo that type casts the yml file loaded..

        self._config = load_yml(config_path)
        super().__init__()

    def setup(self):
        # load the model
        from models import load_predict_model

        self.model = load_predict_model(
            model_name=self._config["model_name"],
            optimizer=self._config["optimizer"],
            loss_fn=self._config["loss_function"],
            weights=self._config["checkpoint"],
        )

        return super().setup()

    def process(self, element):

        model = self.model
        patch = element["patch"]
        prob = round(float(model(patch).numpy()), 2)
        prediction = 1 if prob > 0.5 else 0
        logging.info(f"Prediction for {element['PLOTID']} is {prob} | {prediction}")
        yield {
            "PLOTID": element["PLOTID"],
            "lon": element["lon"],
            "lat": element["lat"],
            "r50_prob": prob,
            "r50_pred": prediction,
        }


# https://github.com/kubeflow/examples/blob/master/LICENSE
class DictToCSVString(beam.DoFn):
    """Convert incoming dict to a CSV string.

    This DoFn converts a Python dict into
    a CSV string.

    Args:
      fieldnames: A list of strings representing keys of a dict.
    """

    def __init__(self, fieldnames):
        super(DictToCSVString, self).__init__()

        self.fieldnames = fieldnames

    def process(self, element, *_args, **_kwargs):
        """Convert a Python dict instance into CSV string.

        This routine uses the Python CSV DictReader to
        robustly convert an input dict to a comma-separated
        CSV string. This also handles appropriate escaping of
        characters like the delimiter ",". The dict values
        must be serializable into a string.

        Args:
          element: A dict mapping string keys to string values.
            {
              "key1": "STRING",
              "key2": "STRING"
            }

        Yields:
          A string representing the row in CSV format.
        """
        fieldnames = self.fieldnames
        filtered_element = {
            key: value for (key, value) in element.items() if key in fieldnames
        }
        logging.info(f"Writing {filtered_element} to CSV")
        with io.StringIO() as stream:
            writer = csv.DictWriter(stream, fieldnames)
            writer.writerow(filtered_element)
            csv_string = stream.getvalue().strip("\r\n")

        yield csv_string


def pipeline(dotargs: SimpleNamespace):
    import time
    st = time.time()

    pColl = parse_shp_to_latlon(dotargs.input)
    cols = ["PLOTID", "lon", "lat", "r50_prob", "r50_pred"]
    
    # TODO: difficulty constructing PipelineOptions obj correctly to pass to beam.Pipeline()
    # this affects ability to host on Dataflow with DataflowRunner vs locally with DirectRunner
    # options = PipelineOptions(['--runner', 'Direct',
    #                            '--direct_num_workers', 32,
    #                            '--direct_running_mode', 'multi_processing']
    #                            )
    # options = PipelineOptions(
    #                             direct_num_workers=32,
    #                             direct_running_mode="multi_processing"
    #                            )
    # options = PipelineOptions(runner=dotargs.runner,
    #                         direct_num_workers=dotargs.direct_num_workers,
    #                         direct_running_mode=dotargs.direct_running_mode
    #                         )
    # print(options.get_all_options())
    with beam.Pipeline() as p: # no options till i figure out how to pass them to beam.Pipeline() 
        forest_pipeline = (
            p
            | "Construct PCollection" >> beam.Create(pColl)
            | "Get Patch" >> beam.ParDo(GetPatch())
            | "Predict" >> beam.ParDo(Predict(config_path=dotargs.model_config))
            | "Dict To CSV String" >> beam.ParDo(DictToCSVString(cols))
            | "Write String To CSV" >> WriteToText(dotargs.output, header=",".join(cols))
        )
        print()
    
    logging.info(f"Pipeline completed in {time.time()-st} seconds")

def run():
    argparse.FileType()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--model-config", "-mc", type=str, required=True)
    
    group = parser.add_argument_group("pipeline-options")
    group.add_argument("--runner", "-r", type=str, required=False)
    group.add_argument("--direct-num-workers", "-d", type=int, required=False)
    group.add_argument("--direct-running-mode", "-m", type=str, required=False)
    args = parser.parse_args()
    print(args.runner)
    print(args.direct_num_workers)
    print(args.direct_running_mode)
    pipeline(dotargs=args)
    
    logging.info(f"merging outputs to one dataframe")
    _parent = Path(args.output).parent
    files = [(_parent/ file) for file in os.listdir(_parent) if file.startswith(Path(args.output).stem)]
    logging.info(f"Merging {len(files)} files")

    # merge all .csv shard files
    df = pd.concat([pd.read_csv(file) for file in files])
    logging.info(f"joining model predictions with input shapefile: {args.input}")
    
    # join it with the input shapefile 
    shp = gpd.read_file(args.input)
    shp['PLOTID'] = shp['PLOTID'].astype('int64')
    # inner join the df with args.input on the PLOTID column
    joined = shp.join(df.set_index('PLOTID'), on='PLOTID')

    # save the geodataframe as a shapefile
    logging.info(f"writing merged shapefile to: {args.output}")
    joined.to_file(args.output)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
