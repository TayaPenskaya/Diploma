# Pose-Transfer

## Requirement
* pytorch(1.6.0)
* torchvision(0.7.0)
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate


### Data Preperation

#### DeepFashion

- Download [deep fasion dataset in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). You will need to ask a pasword from dataset maintainers. 
- Split the raw images into the train split (```fashion_data/train```) and the test split (```fashion_data/test```). Crop the images. Launch
```bash
python tool/generate_fashion_datasets.py
``` 
- Download train/test pairs and train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg), including **fasion-resize-pairs-train.csv**, **fasion-resize-pairs-test.csv**, **fasion-resize-annotation-train.csv**, **fasion-resize-annotation-train.csv**. Put these four files under the ```fashion_data``` directory.
- Generate the pose heatmaps. Launch
```bash
python tool/generate_pose_map_fashion.py
```
- Generate body segements. Launch
```bash
python tool/gen_segment_bbox.py
```

#### Notes:

1. Keypoints files

We use [OpenPose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) to generate keypoints. 

- Download pose estimator from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg). Put it under the root folder ``Pose-Transfer``.
- Change the paths **input_folder**  and **output_path** in ``tool/compute_coordinates.py``. And then launch
```bash
python2 compute_coordinates.py
```

2. Dataset split files

```bash
python2 tool/create_pairs_dataset.py
```

<!-- #### Pose Estimation
- Download the pose estimator from [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).
- Launch ```python compute_cordinates.py``` to get the pose estimation for both datasets. --> 

### Train a model

DeepFashion
```bash
python train.py --dataroot ./fashion_data/ --name fashion_PATN_Fine --model PATN_Fine --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --lambda_style 10 --dataset_mode key_segments --n_layers 3 --norm instance --batchSize 4 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN_Fine --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_seperate_segments_style --which_model_netD resnet_in --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0 --no_lsgan
```


### Test the model

DeepFashion
```bash
python test.py --dataroot ./fashion_data/ --name fashion_PATN_Fine --model PATN_Fine --phase test --dataset_mode key_segments --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN_Fine --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
```


### Pre-trained model 
Our pre-trained model can be downloaded [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg).
