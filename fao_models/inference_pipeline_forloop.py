# testing
import argparse
import ee
import google.auth
import yaml
import beam_utils
from shapely.geometry import Point
import geopandas as gpd
import tqdm
def main():
    PROJECT = "pc530-fao-fra-rss"  # change to your cloud project name

    ## INIT WITH HIGH VOLUME ENDPOINT
    credentials, _ = google.auth.default()
    ee.Initialize(
        credentials,
        project=PROJECT,
        opt_url="https://earthengine-highvolume.googleapis.com"
    )
    
    # parser = argparse.ArgumentParser(description="Run inference on Sentinel-2 images.")
    # parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file.")
    # parser.add_argument("-f", "--file", type=str, required=True, help="Path to the shapefile.")
    
    # args = parser.parse_args()
    # -c runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml -f /home/kyle/Downloads/hexagons_NEW_Brazil_test_subset.shp
    
    config = 'runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml'#args.config
    # shapefile_path = '/home/kyle/Downloads/hexagons_NEW_Brazil_test_subset.shp'#args.file
    shapefile_path = '/home/kyle/Downloads/Brazil_update_nonforests_ssl4eo_fps_removefields_fixPLID.shp'#args.file

    
    if isinstance(config, str):
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    model_name = config["model_name"]
    optimizer = config["optimizer"]
    loss_function = config["loss_function"]
    weights = config["checkpoint"]
    
    # load model
    model = beam_utils.load_predict_model(model_name, optimizer, loss_function, weights)

    # pColl = beam_utils.parse_shp_to_latlon('/home/kyle/Downloads/Brazil_update_hex_test_subset.shp')
    # pColl = beam_utils.parse_shp_to_latlon('/home/kyle/Downloads/hexagons_NEW_Brazil_test_subset.shp')
    pColl = beam_utils.parse_shp_to_latlon(shapefile_path)
    plotids = []
    coords = []
    probs = []
    preds = []
    for nested_l in tqdm.tqdm(pColl):
        plotid = nested_l[0]
        coord = nested_l[1]
        img = beam_utils.get_ee_img(coord)
        patch = beam_utils.get_patch_numpy(coord, img)
        tensor = beam_utils.to_tensor(patch)
        prediction = beam_utils.make_inference(model,tensor)
        plotids.append(plotid)
        coords.append(coord)
        probs.append(prediction[0])
        preds.append(prediction[1])
    print(coords)
    print(preds)
    # Create a GeoDataFrame from the coords and preds
    geometry = [Point(coord) for coord in coords]
    gdf = gpd.GeoDataFrame({'PLOTID':plotids,
                            'r50_prob':probs,
                            'r50_pred':preds}, 
                            geometry=geometry, 
                            )

    # Save the GeoDataFrame to a new shapefile
    output_path = f"{shapefile_path}_resnet_predict.shp"
    gdf.to_file(output_path)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()