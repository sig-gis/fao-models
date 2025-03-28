from pathlib import Path
import tqdm
import subprocess
import argparse
import os
import geopandas as gpd

def main():
    parser = argparse.ArgumentParser(description="Batcher for running multiple beam_pipeline.py instances")
    
    # I/O args
    parser.add_argument('--inputs', 
                        type=str, 
                        required=True, 
                        help='Path to the input text file containing shapefiles'
                        )
    parser.add_argument('--out_dir', 
                        type=str, 
                        required=True, 
                        help='Output directory for the processed shapefiles'
                        )
    
    # Forest/Non-Forest args
    parser.add_argument('--fnf-config', 
                        type=str, 
                        required=True, 
                        default='./_config/runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml', 
                        help='Path to the Forest/NonForest model configuration file'
                        )

    # Change Detection args
    parser.add_argument('--cd-configs', 
                        type=str, 
                        required=True, 
                        default='./configs/', 
                        help='Path to the change-detection configs directory'
                        )
    parser.add_argument('--cd-model', 
                        type=str, 
                        required=True, 
                        default='prithvi', 
                        help='base encoder model for change detection model'
                        )
    parser.add_argument('--cd-weights', 
                        type=str, 
                        required=True, 
                        default='./model/checkpoint__best.pth', 
                        help='Path to the change-detection model checkpoint weights'
                        )
    parser.add_argument('--cd-t1start', 
                        type=str, 
                        required=False, 
                        default= '2018-01-01',
                        help='Start date for the first time period for change detection'
                        )
    parser.add_argument('--cd-t1end', 
                        type=str, 
                        required=False, 
                        default= '2018-12-31',
                        help='End date for the first time period for change detection'
                        )
    parser.add_argument('--cd-t2start', 
                        type=str, 
                        required=False, 
                        default= '2023-01-01',
                        help='Start date for the second time period for change detection'
                        )
    parser.add_argument('--cd-t2end', 
                        type=str, 
                        required=False, 
                        default= '2023-12-31',
                        help='End date for the second time period for change detection'
                        )
  
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_txt = Path(args.inputs).resolve()
    # config = Path(args.config).resolve()
    output_root = Path(out_dir)
   

    with open(inputs_txt) as f:
        shps = f.readlines()
        shps = [shp.rstrip('\n') for shp in shps]

    for input_shp in tqdm.tqdm(shps):
        input_shp_path = Path(input_shp).resolve()
        print(f"input_shp: {input_shp_path}")
        
        input_shp_with_fnf_suffix = input_shp_path.stem + "_fnf.shp"
        out_fnf_shp = output_root / input_shp_with_fnf_suffix
        print(f"output fnf shp: {out_fnf_shp}")

        input_shp_with_cd_suffix = input_shp_path.stem + "_cd.shp"
        out_cd_shp = output_root / input_shp_with_cd_suffix
        print(f"output cd shp: {out_cd_shp}")        
        
        # Run FNF model pipeline
        try:
            result = subprocess.run(
                ["python", "beam_pipeline.py", 
                 "--input", str(input_shp_path), 
                 "--output", str(out_fnf_shp), 
                 "--model-config", str(args.fnf_config)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_shp_path}: {e.stderr}")
        
        # Run Change Detection model pipeline
        try:
        
            result = subprocess.run(
                ["python", "cd_inference_pipeline.py", 
                    "--shapefile", str(input_shp_path), 
                    "--model", str(args.cd_model), 
                    "--outfile", str(out_cd_shp),
                    "--configs", str(args.cd_configs),
                    "--weights", str(args.cd_weights),
                    "--t1start", str(args.cd_t1start),
                    "--t2start", str(args.cd_t2start),
                    "--t1end", str(args.cd_t1end),
                    "--t2end", str(args.cd_t2end),
                 ],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_shp_path}: {e.stderr}")
    
        # merge both model inference shps together and save out
        print(f"Merging: {out_fnf_shp}\n {out_cd_shp}")
        # merge the two shapefiles
        out_fnf_df = gpd.read_file(out_fnf_shp)
        out_fnf_df.loc[:,'PLOTID'] = out_fnf_df['PLOTID'].astype(int)

        out_cd_df = gpd.read_file(out_cd_shp)
        out_cd_df.loc[:,'PLOTID'] = out_cd_df['PLOTID'].astype(int)

        both_models_df = out_fnf_df.merge(out_cd_df,left_on='PLOTID',right_on='PLOTID')
        both_models_df = both_models_df.drop(columns=['SAMPLEID_y','geometry_y'])
        both_models_df = both_models_df.rename(columns={'SAMPLEID_x':'SAMPLEID','geometry_x':'geometry'})
        
        out_final_file = output_root / f"{input_shp_path.stem}_both_models.shp"
        both_models_df.to_file(out_final_file)
        print(f"Merged the two shapefiles, saving to {out_final_file}")
        # remove the individual shapefiles
        for shp in [out_fnf_shp, out_cd_shp]:
            os.remove(shp)


if __name__ == "__main__":
    main()