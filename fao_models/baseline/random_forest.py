from pathlib import Path
from typing import Optional
import ee
import yaml
from dataclasses import dataclass
from datetime import datetime

from indices import add_features, bnin


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
    init_args: dict


@dataclass
class ImageInfo:
    type: str
    features: list[str]
    start_date: Optional[str | datetime] = None
    end_date: Optional[str | datetime] = None
    cloud_mask: Optional[bool] = False


@dataclass
class TrainInfo:
    src: str | Path
    pre_computed_samples: bool
    model_dest: str | Path
    features: Optional[list[str]] = None


@dataclass
class EvaluationInfo:
    src_model: str | Path
    output: str | Path


def default_s2_composite(start_date: datetime, end_date: datetime, features: list[str]):
    # TODO fix later, loading from yml converts to datetime which should work but isn't in filterDate
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(
        ee.Filter.neq("system:index", "20190417T065631_20190417T070736_T38NRL")
    )
    # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
    # Level 1C data and can be applied to either L1C or L2A collections.
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filter(
        ee.Filter.neq("system:index", "20190417T065631_20190417T070736_T38NRL")
    )
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
    image = image.multiply(0.0001)
    return image


def make_composite(**kwargs):
    return default_s2_composite(**kwargs)


def sample_image(image: ee.Image, samples: ee.FeatureCollection, **kwargs):
    # _samples = image.reduceRegions(
    #     collection=samples, reducer=ee.Reducer.first(), scale=200, **kwargs
    # )
    _samples = image.sampleRegions(collection=samples, scale=20, **kwargs)
    _samples = _samples.map(lambda f: f.setGeometry(ee.Geometry.Point([0, 0])))
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


def export_samples(_input, dev):
    cnfg = load_yml(_input)
    image_cnfg = ImageInfo(**cnfg["image"])
    samples = ee.FeatureCollection(cnfg["training_data"]).merge(
        ee.FeatureCollection(cnfg["training_data"])
    )
    output = cnfg["sampling"]["output"]
    if dev:
        samples = samples.limit(10_000)
    img = get_image(image_cnfg)

    # sample imagery
    samples_with_feats = sample_image(img, samples, tileScale=16)

    task = ee.batch.Export.table.toAsset(samples_with_feats, "export-samples", output)
    task.start()
    print("export started:", output)


def train(_input, dev: bool = False):
    # unpack yaml
    cnfg = load_yml(_input)
    classifier_cnfg = ClassifierInfo(**cnfg["classifier"])
    training_cnfg = TrainInfo(**cnfg["training"])

    samples = ee.FeatureCollection(training_cnfg.src)
    if dev:
        samples = samples.limit(1000)

    # sample imagery
    if training_cnfg.pre_computed_samples:
        features = training_cnfg.features
        samples_with_feats = samples.filter(ee.Filter.notNull(features))
    else:
        image_cnfg = ImageInfo(**cnfg["image"])
        features = image_cnfg.features
        img = get_image(image_cnfg)
        samples_with_feats = sample_image(img, samples)
    # build classifer
    model = get_model(
        name=classifier_cnfg.name,
        **classifier_cnfg.init_args,
    )
    # train model
    model = model.train(
        features=samples_with_feats,
        inputProperties=features,
        classProperty=cnfg["target_property"],
    )
    # save model
    #  how to save oob and importance? double check it's not in output
    # explain = model.explain()
    # model.set(
    #     {
    #         "importance": explain.get("importance"),
    #         "outOfBagErrorEstimate": explain.get("outOfBagErrorEstimate"),
    #     }
    # )
    task = ee.batch.Export.classifier.toAsset(
        model, "export-model", training_cnfg.model_dest
    )
    task.start()
    print("started export:", training_cnfg.model_dest)


def evaluate(_input, dev: bool = False):
    cnfg = load_yml(_input)
    image_cnfg = ImageInfo(**cnfg["image"])
    eval_cnfg = EvaluationInfo(**cnfg["evaluation"])

    load = ee.Classifier.load(eval_cnfg.src_model)
    print("loaded:", eval_cnfg.src_model)

    samples = ee.FeatureCollection(cnfg["validation_data"])
    if dev:
        total_size = 100
        s1 = samples.filter("label == 0").limit(total_size // 2)
        s2 = samples.filter("label == 1").limit(total_size // 2)
        samples = ee.FeatureCollection([s1, s2]).flatten()

    img = get_image(image_cnfg)
    samples_with_feats = sample_image(img, samples, tileScale=12)
    samples_with_feats = samples_with_feats.filter(
        ee.Filter.notNull(["B4", "B3", "B2", "B8", "Nstr", "GEZ"])
    )
    predictions = samples_with_feats.classify(load)
    error_matrix = predictions.errorMatrix(
        actual=cnfg["target_property"], predicted="classification"
    )
    if dev:
        print(error_matrix.getInfo())
        print("Overall accuracy:", error_matrix.accuracy().getInfo())
        print("Consumer's accuracy:")
        print(error_matrix.consumersAccuracy().getInfo())
        print("Producer's accuracy:")
        print(error_matrix.producersAccuracy().getInfo())
    else:
        output = eval_cnfg.output
        md = ee.Dictionary(eval_cnfg.__dict__).set("features", load.schema())
        predictions = predictions.set(md)
        task = ee.batch.Export.table.toAsset(predictions, "export-predictions", output)
        task.start()
        print("export started:", output)


if __name__ == "__main__":
    ee.Initialize(project="pc530-fao-fra-rss")
    dev = False
    config = "fao_models/baseline/config_baseline_v0.yml"
    # export_samples(config, dev=dev)
    # train(config, dev=dev)
    evaluate(config, dev=dev)
