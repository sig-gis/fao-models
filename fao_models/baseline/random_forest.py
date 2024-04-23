from pathlib import Path
from typing import Optional
import ee
import ee.batch
import yaml
from dataclasses import dataclass
from datetime import datetime


def load_yml(_input: Path | str):
    # TODO mv to common.py refactor other scripts to use
    with open(_input, "r") as f:
        args = yaml.safe_load(f)

    # #tests for later maybe
    # assert a1 == a2, "PAth and str are not same"
    return args


@dataclass
class ClassifierInfo:
    name: str
    dest: str | Path
    init_args: dict
    train_args: dict


@dataclass
class ImageInfo:
    type: str
    features: list[str]
    start_date: Optional[str | datetime] = None
    end_date: Optional[str | datetime] = None
    cloud_mask: Optional[bool] = False


def default_s2_composite(start_date: datetime, end_date: datetime, features: list[str]):
    # TODO fix later, loading from yml converts to datetime which should work but isn't in filterDate
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
    # Level 1C data and can be applied to either L1C or L2A collections.
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    # Use 'cs' or 'cs_cdf', depending on your use case; see docs for guidance.
    QA_BAND = "cs_cdf"
    # The threshold for masking; values between 0.50 and 0.65 generally work well.
    # Higher values will remove thin clouds, haze & cirrus shadows.
    CLEAR_THRESHOLD = 0.50
    image = (
        s2.filterDate(start_date, end_date)
        .linkCollection(csPlus, [QA_BAND])
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
        .select(features)
        .median()
    )
    return image


def make_composite(**kwargs):
    return default_s2_composite(**kwargs)


def sample_image(image: ee.Image, samples: ee.FeatureCollection, **kwargs):
    _samples = image.reduceRegions(
        collection=samples, reducer=ee.Reducer.first(), scale=20, **kwargs
    )
    return _samples


def get_image(config: ImageInfo):
    # prep imagery
    if config.type == "composite":
        img = make_composite(
            start_date=config.start_date,
            end_date=config.end_date,
            features=config.features,
        )
    else:
        # TODO support image from path
        raise NotImplementedError
    return img


def get_model(name, **kwargs):
    _models = {"randomforest": ee.Classifier.smileRandomForest}
    model = _models.get(name, None)
    if model is not None:
        model = model(**kwargs)
    return model


def train(_input, dev: bool = False):
    # unpack yaml
    cnfg = load_yml(_input)
    image_cnfg = ImageInfo(**cnfg["image"])
    classifier_cnfg = ClassifierInfo(**cnfg["classifier"])
    samples = ee.FeatureCollection(cnfg["training_data"])
    if dev:
        samples = samples.limit(100)
    img = get_image(image_cnfg)

    # sample imagery
    samples_with_feats = sample_image(img, samples)
    # build classifer
    model = get_model(
        name=classifier_cnfg.name,
        **classifier_cnfg.init_args,
    )
    # train model
    model = model.train(
        features=samples_with_feats,
        inputProperties=image_cnfg.features,
        **classifier_cnfg.train_args,
    )
    # save model
    task = ee.batch.Export.classifier.toAsset(
        model, "export-model", classifier_cnfg.dest
    )
    task.start()


def evaluate(_input, dev: bool = False):
    cnfg = load_yml(_input)
    image_cnfg = ImageInfo(**cnfg["image"])
    classifier_cnfg = ClassifierInfo(**cnfg["classifier"])
    load = ee.Classifier.load(classifier_cnfg.dest)
    samples = ee.FeatureCollection(cnfg["testing_data"])
    if dev:
        samples = samples.limit(100)
    img = get_image(image_cnfg)
    samples_with_feats = sample_image(img, samples)
    predictions = samples_with_feats.classify(load)
    error_matrix = predictions.errorMatrix(
        actual=classifier_cnfg.train_args["classProperty"], predicted="classification"
    )
    print("Overall accuracy:", error_matrix.accuracy().getInfo())
    print("Consumer's accuracy:")
    print(error_matrix.consumersAccuracy().getInfo())
    print("Producer's accuracy:")
    print(error_matrix.producersAccuracy().getInfo())


if __name__ == "__main__":
    ee.Initialize(project="pc530-fao-fra-rss")
    dev = False
    train("fao_models/baseline/config.yml", dev=dev)
    # evaluate("fao_models/baseline/config.yml", dev=dev)
