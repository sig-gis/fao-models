from pathlib import Path
import tqdm
import subprocess
import argparse
import os
import geopandas as gpd
import shutil

def main():
    parser = argparse.ArgumentParser(description="Runs FNF and CD inference pipelines for each input SHP listed in the --inputs text file")
    
    # I/O args
    parser.add_argument('--inputs', 
                        type=str, 
                        required=True, 
                        help='Path to the input text file containing shapefiles'
                        )
    
    parser.add_argument('--config', 
                        type=str, 
                        required=True, 
                        default='configs/inference23.yml', 
                        help='Path to the inference config file'
                        )

    parser.add_argument('--cleanup',
                        action='store_true', 
                        help='Remove intermediate files after processing'
                        )
    parser.add_argument('--sa-key',
                        type=str, 
                        required=False, 
                        help='Path to the service account key file if running on a remote machine'
                        )
    
    args = parser.parse_args()

    if args.sa_key: # setting the os.environ key will allow google.auth.default() to initilze EE w service account creds
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.sa_key
        print(f"Using service account key: {args.sa_key}")
        
    # otherwise check that app default creds will be found at auth time
    else:
        ee_config_dir = Path.home() / ".config" / "earthengine" / "credentials"
        try:
            print(f"Will use app default creds found at {ee_config_dir}")

        except FileNotFoundError as e:
            print(e)
            print("Please run `gcloud auth login` to set up your Earth Engine credentials.")
    print('Working directory:', os.getcwd())
    inputs_txt = Path(args.inputs).resolve()

    with open(inputs_txt) as f:
        shps = f.readlines()
        shps = [shp.rstrip('\n') for shp in shps]

    for input_shp in tqdm.tqdm(shps):

        out_fnf_shp = input_shp.replace(".shp","_fnf.shp")
        out_cd_shp = input_shp.replace(".shp", "_cd.shp")
        out_final_file =  input_shp.replace(".shp","_both_models.shp")
        input_shp, out_fnf_shp, out_cd_shp, out_final_file = [Path(path).absolute() for path in [input_shp,out_fnf_shp, out_cd_shp, out_final_file]] 
        
        print(f"input_shp: {input_shp}")
        print(f"output fnf shp: {out_fnf_shp}")
        print(f"output cd shp: {out_cd_shp}")     
        print(f"output final shp: {out_final_file}")

        # # Run FNF model pipeline
        if os.path.exists(out_final_file):
            print(f"Skipping processing for {input_shp} as it has already been processed.")
            continue
        else:
            try:
                result = subprocess.run(
                    ["python", "fao_models/beam_pipeline.py", 
                    "--input", str(input_shp), 
                    "--output", str(out_fnf_shp), 
                    "--model-config", str(args.config)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_shp} through FNF model: {e.stderr}")
        
            try:
                result = subprocess.run(
                    ["python", "fao_models/cd_inference_pipeline.py", 
                        "--input", str(input_shp), 
                        "--output", str(out_cd_shp), 
                        "--config", str(args.config),
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_shp} through CD model: {e.stderr}")
    
            # merge both model inference shps together and save out
            out_fnf_df = gpd.read_file(out_fnf_shp)
            out_fnf_df.loc[:,'PLOTID'] = out_fnf_df['PLOTID'].astype(int)

            out_cd_df = gpd.read_file(out_cd_shp)
            out_cd_df.loc[:,'PLOTID'] = out_cd_df['PLOTID'].astype(int)

            both_models_df = out_fnf_df.merge(out_cd_df,left_on='PLOTID',right_on='PLOTID')
            both_models_df = both_models_df.drop(columns=['SAMPLEID_y','geometry_y'])
            both_models_df = both_models_df.rename(columns={'SAMPLEID_x':'SAMPLEID','geometry_x':'geometry'})
            
            both_models_df.to_file(out_final_file)
            print(f"Merged the two shapefiles, saving to {out_final_file}")
            
            # remove the individual shapefiles
            if args.cleanup:
                print(f"Cleaning up intermediate files for {input_shp}")
                parent = os.path.dirname(input_shp)
                to_delete = [f for f in os.listdir(parent) if "fnf" in f or "cd" in f]
                print(f'removing intmd files: {to_delete}')
                for f in to_delete:
                    try:
                        os.remove(os.path.join(parent,f))
                    except IsADirectoryError:
                        shutil.rmtree(os.path.join(parent,f))
            


if __name__ == "__main__":
    main()