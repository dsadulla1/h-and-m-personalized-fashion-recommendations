# Azure Standard E8bds v5 (8 vcpus, 64 GiB memory) VM
# West US 2
# Data Science Virtual Machine Linux (ubuntu 20.04)

# git config setup
conda init bash
chmod 600 /home/deepaksadulla/.kaggle/kaggle.json
conda create -n Dev python=3.9
conda activate Dev
pip3 install sklearn numpy pandas tensorflow-cpu kaggle xgboost
kaggle competitions download -c h-and-m-personalized-fashion-recommendations