conda activate base

conda create -y -n moa python==3.8.10 

conda activate moa 

conda env update -y -f env.yml 