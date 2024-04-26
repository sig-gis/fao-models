import ee

ee.Initialize(project="pc530-fao-fra-rss")


dem = ee.Image("NASA/NASADEM_HGT/001")
norm_diffs = [
    ["blue", "green"],
    ["blue", "red"],
    ["blue", "nir"],
    ["blue", "swir1"],
    ["blue", "swir2"],
    ["green", "red"],
    ["green", "nir"],
    ["green", "swir1"],
    ["green", "swir2"],
    ["red", "swir1"],
    ["red", "swir2"],
    ["nir", "red"],
    ["nir", "swir1"],
    ["nir", "swir2"],
    ["swir1", "swir2"],
]

ratios = [["swir1", "nir"], ["red", "swir1"]]
# ratios = [["red", "blue"]]

bnread = [
    "Aerosols",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "rededge4",
    "watervapor",
    "swir1",
    "swir2",
]
bnin = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# bnin = ["B2", "B3", "B4", "B5"]


def vector_nd(f, cols):
    cols = ee.List(cols)
    nd = ee.Number.expression(
        "(A - B) / (A + B)", {"A": f.get(cols.get(0)), "B": f.get(cols.get(1))}
    )
    return nd


def vector_ratio(f, cols):
    cols = ee.List(cols)
    ratio = ee.Number.expression(
        "(A) / (B)", {"A": f.get(cols.get(0)), "B": f.get(cols.get(1))}
    )
    return ratio


def calc_vector_indices(f):
    _f = f.select(bnin, bnread)
    norm_diff = ee.List(norm_diffs)
    norm_diffs_vals = norm_diff.map(lambda cols: vector_nd(_f, cols))
    norm_diffs_keys = norm_diff.map(
        lambda cols: ee.List(
            ["nd_", ee.List(cols).get(0), "_", ee.List(cols).get(1)]
        ).join()
    )

    norm_diffs_dict = ee.Dictionary.fromLists(norm_diffs_keys, norm_diffs_vals)

    ratio = ee.List(ratios)
    ratios_vals = ratio.map(lambda cols: vector_ratio(_f, cols))
    ratios_keys = ratio.map(
        lambda cols: ee.List(
            ["r_", ee.List(cols).get(0), "_", ee.List(cols).get(1)]
        ).join()
    )
    ratios_dict = ee.Dictionary.fromLists(ratios_keys, ratios_vals)
    return f.set(norm_diffs_dict).set(ratios_dict)


def calc_image_indices(image):
    _image = image.select(bnin, bnread)
    norm_diffs_images = ee.Image.cat(
        [_image.normalizedDifference(i).rename(f"nd_{i[0]}_{i[1]}") for i in norm_diffs]
    )
    ratio_images = ee.Image.cat(
        [
            _image.select(i[0]).divide(_image.select(i[1])).rename(f"r_{i[0]}_{i[1]}")
            for i in ratios
        ]
    )
    elevation = dem.select("elevation")
    return ee.Image.cat([image, norm_diffs_images, ratio_images, elevation])


def add_features(input: ee.Image | ee.Feature):
    if isinstance(type(input), ee.Image):
        return calc_image_indices(input)
    elif isinstance(type(input), ee.Feature):
        return calc_vector_indices(input)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    fc = ee.FeatureCollection(
        "projects/pc530-fao-fra-rss/assets/reference/split/training_sample_s2"
    )
    fc = fc.limit(100)
    fc = fc.map(calc_vector_indices)
    print(fc.size())
