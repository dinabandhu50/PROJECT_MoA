# if not activated then activate base
conda activate base

# create conda env and activate it
conda env create -f env.yml

# download data
python src/download_data.py