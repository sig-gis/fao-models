conda activate pangaea-bench

python cd_inference_pipeline.py \
    --shapefile ./data/FRA_plots_DemocraticRepublicoftheCongo/centroid_OLDMang_DemocraticRepublicoftheCongo/centroid_OLDMang_DemocraticRepublicoftheCongo.shp \
    --outfile ./predictions/\
    --configs ./configs/ \
    --model prithvi \
    --weights ./model/checkpoint__best.pth \
    --t1start 2018-01-01 \
    --t1end 2018-12-31 \
    --t2start 2023-01-01 \
    --t2end 2023-12-31

   