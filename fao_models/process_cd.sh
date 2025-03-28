conda activate pangaea-bench

python cd_inference_pipeline.py \
    --shapefile /home/kyle/code_repos/fao-models/data/Angola_geometrías_paraPrediccion/Geometry/Mang/centroids_New_Mang_Angola/centroids_New_Mang_Angola.shp \
    --outfile /home/kyle/code_repos/fao-models/data/Angola_geometrías_paraPrediccion/Geometry/Mang/centroids_New_Mang_Angola/centroids_New_Mang_Angola_cd_preds_rerun1.shp \
    --configs ./configs/ \
    --model prithvi \
    --weights ./model/checkpoint__best.pth \
    --t1start 2018-01-01 \
    --t1end 2018-12-31 \
    --t2start 2023-01-01 \
    --t2end 2023-12-31

   