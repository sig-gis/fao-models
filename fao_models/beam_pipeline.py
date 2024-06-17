import collections
import typing
import argparse
from types import SimpleNamespace
import csv
import io
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromCsv, WriteToText

from beam_utils import parse_shp_to_latlon
from common import load_yml


# want my pipeline to have these general steps

# 1. Read in data from SHP (hexagons were provided as SHP and CSV but CSV has no geom column, centroids only came as SHP file)
# 2. parse data into row-wise elements of (global id, [lon,lat]) - rest of pipeline passes these elements through
# 3. download imagery for each element and convert to a tensor
# 4. load model and run inference on tensor to return prediction value
# 5. write prediction value to new CSV file with (global id, lat, long, prediction value)


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

        image = get_ee_img(coords)
        patch = get_patch_numpy(coords, image)
        patch_tensor = to_tensor(patch)

        yield {
            "id": global_id,
            "long": coords[0],
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
        from models import get_model, freeze

        self.model = get_model(
            model_name=self._config["model_name"],
            optimizer=self._config["optimizer"],
            loss_fn=self._config["loss_function"],
            training_mode=True,
        )
        self.model.load_weights(self._config["checkpoint"])
        freeze(self.model)

        return super().setup()

    def process(self, element):

        model = self.model
        patch = element["patch"]
        prob = round(float(model(patch).numpy()), 2)
        prediction = "Forest" if prob > 0.5 else "Non-Forest"

        yield {
            "id": element["id"],
            "long": element["long"],
            "lat": element["lat"],
            "prob_label": prob,
            "pred_label": prediction,
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
        with io.StringIO() as stream:
            writer = csv.DictWriter(stream, fieldnames)
            writer.writerow(filtered_element)
            csv_string = stream.getvalue().strip("\r\n")

        yield csv_string


def pipeline(beam_options, dotargs: SimpleNamespace):
    if beam_options is not None:
        beam_options = PipelineOptions(**load_yml(beam_options))

    pColl = parse_shp_to_latlon(dotargs.input)
    cols = ["id", "long", "lat", "prob_label", "pred_label"]
    with beam.Pipeline() as p:
        var = (
            p
            | "Construct PCollection" >> beam.Create(pColl)
            | "Get Patch" >> beam.ParDo(GetPatch())
            | "Predict" >> beam.ParDo(Predict(config_path=dotargs.model_config))
            | "Dict To CSV String" >> beam.ParDo(DictToCSVString(cols))
            | "Write String To CSV"
            >> WriteToText(dotargs.output, header=",".join(cols))
        )


# test file
# file = 'C:\\Users\\kyle\\Downloads\\FRA_hex_shp_5records.shp'
def run():
    argparse.FileType()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--model-config", "-mc", type=str, required=True)
    group = parser.add_argument_group("pipeline-options")
    group.add_argument("--beam-config", "-bc", type=str)
    args = parser.parse_args()

    pipeline(beam_options=args.beam_config, dotargs=args)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
