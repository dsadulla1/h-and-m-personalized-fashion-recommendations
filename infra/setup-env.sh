# Azure Standard E8bds v5 (8 vcpus, 64 GiB memory) VM
# West US 2
# Data Science Virtual Machine Linux (ubuntu 20.04)

# git config setup
conda init bash
chmod 600 /home/deepaksadulla/.kaggle/kaggle.json
conda create -n Dev python=3.9
conda activate Dev
pip3 install sklearn numpy pandas tensorflow-cpu kaggle xgboost jupyterlab matplotlib
cd <required folder>
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip -qq h-and-m-personalized-fashion-recommendations.zip
chmod 600 /home/deepaksadulla/.kaggle/kaggle.json
kaggle datasets download -d deepaksadulla/hm-image-features-w-resnet50
kaggle datasets download -d deepaksadulla/hm-text-features-w-roberta
unzip -qq hm-text-features-w-roberta.zip
unzip -qq hm-image-features-w-resnet50.zip