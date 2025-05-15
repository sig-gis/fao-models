# testing
import os
from pathlib import Path
import argparse
import ee
import google.auth
import yaml
import geopandas as gpd
import tqdm

import argparse
import logging
import datetime

import torch

import numpy as np
import pandas as pd

from model.util import build_change_detection_model
from cd_utils import transform,get_landsat_composite, get_arr_from_geom_centr

logging.basicConfig(
    filename=f"change-detection-{datetime.datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}.log",
    encoding="utf-8",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)

means = np.array([0.05278337,0.08498019,0.10346901,0.2802707,0.25964622 ,0.16640756])
stds = np.array([0.03278688, 0.05424733 ,0.08996119 ,0.07969411 ,0.12222017 ,0.12167657])
classes ={
    'Stable Non Forest': 0,
    'Stable Forest': 1,
    'Forest Loss': 2,
    'Forest Gain': 3,
}

PROJECT = "pc530-fao-fra-rss"  # change to your cloud project name

# INIT WITH HIGH VOLUME ENDPOINT
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
ee.Initialize(
    credentials,
    project=PROJECT,
    opt_url="https://earthengine-highvolume.googleapis.com"
)
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ.keys():
            logging.info(f"Earth Engine initialized at subprocess level with {credentials.signer_email}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", "-s", type=str, required=True)
    parser.add_argument('--model','-m',type=str,required=True)
    parser.add_argument("--outfile", "-o", type=str, required=True)
    parser.add_argument("--configs", "-c", type=str, required=True)
    parser.add_argument("--weights",'-w',type=str,required=True)
    parser.add_argument('--t1start',type=str,required=True)
    parser.add_argument('--t2start',type=str,required=True)
    parser.add_argument('--t1end',type=str,required=True)
    parser.add_argument('--t2end',type=str,required=True)
    
    args = parser.parse_args()

    
    gsd = 10
    size = 32
    
    shapefile_path = args.shapefile
    features = gpd.read_file(shapefile_path)

    with open(os.path.join(args.configs,f'{args.model}.yaml'),'r') as encoder_file:
        encoder_config = yaml.safe_load(encoder_file)
    with open(os.path.join(args.configs,'cls_linear_mt_ltae.yaml'),'r') as decoder_file:
        decoder_config = yaml.safe_load(decoder_file)
    
    # load model
    model = build_change_detection_model(encoder_config,decoder_config,args.weights)

    plotids = []
    preds = []
    confs = []
    for i in tqdm.tqdm(transform(features,args.t1start,args.t1end,args.t2start,args.t2end)):

        plotid = i['PLOTID']
        geometry = i['geometry']

        sample_dates = i['sample_dates']
        logging.info(f"Getting Imagery for {plotid} at {geometry}")

        image1 = get_landsat_composite(region=geometry, start=sample_dates['t1start'], end=sample_dates['t1end'])
        image2 = get_landsat_composite(region=geometry, start=sample_dates['t2start'], end=sample_dates['t2end'])



        arr1 = get_arr_from_geom_centr(image=image1, geom=geometry, gsd=gsd, size=size)
        arr1_normed = (arr1 - means[:,None,None]) / stds[:,None,None]
        # print(arr1_normed)


        arr2 = get_arr_from_geom_centr(image=image2, geom=geometry, gsd=gsd, size=size)
        arr2_normed = (arr2 - means[:,None,None]) / stds[:,None,None]
        # print(arr2_normed)

        arr1_normed = arr1_normed.astype(np.float32)
        arr2_normed = arr2_normed.astype(np.float32)

        input = torch.from_numpy(np.stack([arr1_normed,arr2_normed]))
        input = input.unsqueeze(0)
        input = torch.permute(input,(0,2,1,3,4))
        
        pred = model({'optical':input})
        pred_index = np.argmax(pred.detach().numpy())

        cls = list(classes.keys())[pred_index]
        # print(plotid)
        # print(pred)
        # print(cls)
        # print(pred_index)
    

        plotids.append(plotid)
        preds.append(cls)
        confs.append(pred.detach().numpy())
        logging.info(f"PLOTID: {plotid} Prediction: {cls} Confidence: {pred.detach().numpy()}")


    confs_matrix = np.concatenate(confs,axis=0)
    # print(confs_matrix.shape)

    preds = pd.DataFrame.from_dict({
        'PLOTID':plotids,
        'CD_Pred':preds,
        'SNF_Conf':confs_matrix[:,0],
        'SF_Conf':confs_matrix[:,1],
        'FL_Conf':confs_matrix[:,2],
        'FG_Conf':confs_matrix[:,3]
    })

    joined = features.merge(preds,on='PLOTID')
    
    Path(args.outfile).parent.mkdir(parents=True,exist_ok=True)
    joined.to_file(args.outfile,driver='ESRI Shapefile')
    
    logging.info(f'Predictions Saved to {args.outfile}')

if __name__ == "__main__":
    main()