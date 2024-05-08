# install mambda
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash yes | Miniforge3-$(uname)-$(uname -m).sh


conda create -n torch python=3.11 -y
conda activate torch

mamba install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia