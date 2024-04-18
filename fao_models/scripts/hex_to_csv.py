import ee
import requests

PROJECT = "pc530-fao-fra-rss"
ee.Initialize(project=PROJECT)

hex = ee.FeatureCollection(
    "projects/pc530-fao-fra-rss/assets/reference/hexWCenPropertiesTropics"
)
hex_forest = hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 1), ee.Filter.eq("LU18CEN", "Forest"))
)
hex_nonforest = hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 0), ee.Filter.neq("LU18CEN", "Forest"))
)
hex_samples = (
    ee.FeatureCollection([hex_forest, hex_nonforest]).flatten().randomColumn(seed=42)
)


# .Point(lng, lat).
def to_centroid(feat: ee.Feature):
    feat = feat.centroid()
    coords = feat.geometry().coordinates()
    return feat.set({"lng": coords.get(0), "lat": coords.get(1)})


hex_samples = hex_samples.map(to_centroid)
print(hex_samples.limit(1).getInfo())
# split into train test and validate 70, 20, 10
training_sample = hex_samples.filter("random <= 0.7")
testing_sample = hex_samples.filter("random > 0.7 and random <=.9")
validation_sample = hex_samples.filter("random > 0.9")
# print(
#     training_sample.size().getInfo(),
#     testing_sample.size().getInfo(),
#     validation_sample.size().getInfo(),
# )


def download(collection: ee.FeatureCollection, full_namepath: str):
    url = collection.getDownloadURL()
    response = requests.get(url)
    response.raise_for_status()
    with open(full_namepath, "wb") as fd:
        fd.write(response.content)


download(training_sample, "training_sample.csv")
download(testing_sample, "testing_sample.csv")
download(validation_sample, "validation_sample.csv")
