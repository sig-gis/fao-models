from pathlib import Path
import tqdm
import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Batcher for running multiple beam_pipeline.py instances")
    parser.add_argument('--inputs', type=str, required=True, help='Path to the input text file containing shapefiles')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the processed shapefiles')
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_txt = Path(args.inputs).resolve()
    config = Path(args.config).resolve()
    output_root = Path(out_dir)
    # inputs_txt = "/home/kyle/code_repos/fao-models/batch_runs/inputs_drc.txt"
    # config = "/home/kyle/code_repos/fao-models/_config/runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml"
    # output_root = Path("/home/kyle/Downloads/DRC_outputs")

    with open(inputs_txt) as f:
        shps = f.readlines()
        shps = [shp.rstrip('\n') for shp in shps]

    for input_shp in tqdm.tqdm(shps):
        input_shp_path = Path(input_shp).resolve()
        print(f"input_shp: {input_shp_path}")
        input_shp_with_suffix = input_shp_path.stem + "_r50.shp"
        out_shp = output_root / input_shp_with_suffix
        print(f"output shp: {out_shp}")

        try:
            result = subprocess.run(
                ["python", "fao_models/beam_pipeline.py", "--input", str(input_shp_path), "--output", str(out_shp), "--model-config", str(config)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_shp_path}: {e.stderr}")

if __name__ == "__main__":
    main()