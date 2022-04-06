#!/bin/sh

# usage:
#    set gft appropriately
#    sh create_new_venv.sh <name_of_new_venv>
#    


python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv $HOME/venv/$1

module load python/3.7.9
source $HOME/venv/$1/bin/activate
module load cuda                                                                
module load cudnn                                                               
module load nccl
module load :common-libraries :codecs sndfile
module load r-project
module load ffmpeg
pip install -r requirements.txt


