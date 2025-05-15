# fao-models

Image Classification models for the FAO's Forest Resource Assessment data collection campaigns in Collect Earth Online.

The FAO FRA team is conducting its next round of global data collection surveys. They use Collect Earth Online and local trained interpreters. Quality of interpreter answers have been a concern in the past. Answers between multiple interpreters on the same plot may disagree for a variety of reasons, but there has not been a good way to flag certain plots or interpreters in an informed way. We are developing EO models to act as a 'source of truth' in a new CEO QA/QC feature, so that project admins can identify plots or interpreters who display high levels of disagreement with a modeled 'source of truth'.

This repo contains our two models for production: 
* Forest/Non-Forest model (image classification; CNN architecture; tensorflow framework)
* Forest Change Detection model (image classification; ViT architecture fine-tuned from Prithvi 100M; Pytorch framework)

### Setup

In a new virtual environment, install dependencies using provided requirements.txt when at repo root
`pip install -r requirements.txt`

### Inference

Using [beam_batcher.py](fao_models/beam_batcher.py) allows you to run both models' inference pipelines in succession for each input Shapefile in a `inputs_list.txt`, appending their predictions to one final Shapefile. 

Here is an example:
Run beam_batcher on several input Shapefiles defined in `batches/inputs_test.txt`, output each Shapefile with model predictions to a new directory `test_preds`

```bash
cd fao_models
python beam_batcher.py --inputs batches/inputs_test.txt --out_dir data/inference/test_preds --fnf-config fnf_config/runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml --cd-configs configs/ --cd-model prithvi --cd-weights model/checkpoint__best.pth --cd-t1start 2018-01-01 --cd-t1end 2018-12-31 --cd-t2start 2023-01-01 --cd-t2end 2023-12-31 --cleanup
```

* If running inference with the current models and for inference year 2023, only flags you need to change are `--inputs` and `--out_dir`. 
* See [`batches/inputs_template.txt`](/fao_models/batches/inputs_template.txt) for an example of what to provide to `--inputs`. 
* For FNF model, inference year is hard-coded to 2023. (TODO fix this in [`beam_utils.get_ee_img()`](fao_models/beam_utils.py))
* For CD model, you can change the image time windowing provided to model inference by with the 4 `--cd-t*` flags. 

#### Running on a remote machine with a service account

If you would like to run inference on a remote machine unattended you should use a google cloud service account. You can provide the path to your service account's credentials JSON key with the `--sa-key` flag. 

To use a service account, you first need to create one in Google Cloud ([docs](https://cloud.google.com/iam/docs/service-accounts-create#console)), set its IAM permissions to access Earth Engine ([docs](https://cloud.google.com/iam/docs/grant-role-console); `Earth Engine Resource Writer` and `Service Usage Consumer` should be sufficient), then download a private key to your remote machine ([docs](https://cloud.google.com/sdk/gcloud/reference/iam/service-accounts/keys/create)). 

I have had success running bigger jobs on a [SEPAL](https://sepal.io) `m4` vm with a service account and transferring the outputs to GCS afterward with `gcloud`.

### Further Development

While the state and intent of this repo currently is for inference production requests, all code for training the F/NF Tensorflow model is here (see [`fao_models/model_fit.py`](fao_models/model_fit.py)). Code for fine-tuning the CD model from Prithvi 100M was done in SIG's [pangaea-bench repo fork](https://github.com/sig-gis/pangaea-bench/tree/faofra), but the final model checkpoint and code required to correctly instantiate the model for inference is ported over for simplicity. Reach out to Kyle Woodward (kwoodward@sig-gis.com) and Ryan Demilt (rdemilt@sig-gis.com) for more info. 