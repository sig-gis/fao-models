import os

import numpy as np
import rasterio as rio

DATA = os.path.abspath("data")
filenames = [os.path.join(DATA, f) for f in os.listdir(DATA) if f.endswith(".tif")]

for file in filenames[20_000:]:
    with rio.open(file) as dst:
        msk = dst.read_masks()
        if np.min(msk) == 0:
            print(file)
# files w no data.
# /Users/johndilger/Documents/projects/fao-models/data/patch_pt83312_nonforest.tif
# /Users/johndilger/Documents/projects/fao-models/data/patch_pt42114_nonforest.tif
# /Users/johndilger/Documents/projects/fao-models/data/patch_pt84910_nonforest.tif


# # test np.min on multi array logic
# test = np.zeros((4, 5, 5))
# test = test + 255
# test[3, 3, 3] = 0
# # print(test)
# print('min should be 0 with 1 "masked" pixel: min', np.min(test))
