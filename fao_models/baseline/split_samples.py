import ee


ee.Initialize(project="pc530-fao-fra-rss")
all_points = ee.FeatureCollection(
    "projects/pc530-fao-fra-rss/assets/reference/centroidsTropics"
)
hex_forest = all_points.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 1), ee.Filter.eq("LU18CEN", "Forest"))
).map(lambda f: f.set("label", 1))
hex_nonforest = all_points.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 0), ee.Filter.neq("LU18CEN", "Forest"))
).map(lambda f: f.set("label", 0))
hex_samples = (
    ee.FeatureCollection([hex_forest, hex_nonforest]).flatten().randomColumn(seed=42)
)

training_sample = hex_samples.filter("random <= 0.7")
testing_sample = hex_samples.filter("random > 0.7 and random <=.9")
validation_sample = hex_samples.filter("random > 0.9")

root = "projects/pc530-fao-fra-rss/assets/reference/split"
ee.batch.Export.table.toAsset(
    training_sample, "training_sample", f"{root}/training_sample"
).start()
ee.batch.Export.table.toAsset(
    testing_sample, "testing_sample", f"{root}/testing_sample"
).start()
ee.batch.Export.table.toAsset(
    validation_sample, "validation_sample", f"{root}/validation_sample"
).start()
