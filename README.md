# Multi-Frame-Image-Fusion-using-a-Machine-Learning-based-Weight-Mask-Predictor
This repository was created to present the work presented in Multi-Frame Image Fusion using a Machine Learning based Weight Mask Predictor for Turbulence-Induced Image Degradation (Submited to SPIE Journal of Applied Remote Sensing on 4/6/2022)

**Download HR Image Dataset**

Download the High resolution DIV2K and/or Flickr2K dataset set from https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW 

Add training images to images/Train_HR

Add validation images to images/Valid_HR

**Generate Training and validation Datasets**

Run the MATLAB file: Matlab_scripts/Gen_train_valid.m

Parameters for generating the degraded images can be found on lines 20-44.
Images are saved to images folder.

**Generate TestingDataset**

Run the MATLAB file: Matlab_scripts/Gen_test_full.m

**Train Network**

python Python_scripts/main.py --cuda --mse_avg --test --save_iter=50 --nEpochs=100 --batchSize=10 --trainsize=4000 --testsize=100 --loss=8 --loss_2=4 --loss_2_weight=1 --loss_dis=3 --adv_loss_weight=0.001

**Run and evaluate aginst test dataset**

python Python_scripts/test_eval.py --global_align --local_align --align_avg --cuda --valid_dir=images/full_test --valid_size=100 --model_path=model/Weight_Generator_FFT_correntropy_050Image_Discrminator_WGANGP/epoch_100.pth --output_dir=results_extreme_FFT_correntropty_aligned_avg_128 --input_height=128 --input_width=128

**Runtest againt real underwater turbulence dataset**

python Python_scripts/test_real.py --global_align --local_align --align_avg --cuda --valid_dir=underwater_turbulence --valid_size=3 --model_path=model/Weight_Generator_FFT_correntropy_050Image_Discrminator_WGANGP/epoch_100.pth --y --input_height=256 --input_width=256 output_dir=results_USAF_underwater_FFT_correntropt



