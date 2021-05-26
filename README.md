# PyTorch Implementation of [EigenGAN](https://arxiv.org/pdf/2104.12476.pdf) 


**Train**
```
python train.py [image_folder_path] --name [experiment name]
```


**Test**
```
python test.py [ckpt path] --traverse
```


**FFHQ** 

[[ckpt]](https://drive.google.com/file/d/1WNHlNibrgEo0elVB-epS5GO2J0YzkvNC/view?usp=sharing)

samples (no truncation)

![./docs/ffhq/sample.jpg](./docs/ffhq/sample.jpg)

Learned subspace: L0 D1
![./docs/ffhq/traverse_L0_D1.jpg](./docs/ffhq/traverse_L0_D1.jpg)

Learned subspace: L1 D2
![./docs/ffhq/traverse_L1_D2.jpg](./docs/ffhq/traverse_L1_D2.jpg)


**Anime**

[[ckpt]](https://drive.google.com/file/d/1NO6oXs4yvtIidirXqG9HxbEyHS9l8jdR/view?usp=sharing)

samples (no truncation)

![./docs/anime/sample.jpg](./docs/anime/sample.jpg)

Learned subspace: L0 D5
![./docs/anime/traverse_L0_D5.jpg](./docs/anime/traverse_L0_D5.jpg)

Learned subspace: L1 D2
![./docs/anime/traverse_L1_D2.jpg](./docs/anime/traverse_L1_D2.jpg)



**Note** 

Default training configurations are different from the original implementation 

Tested on python 3.8 + torch 1.8.1


**Issue**

Some of the subspace layers seem to collapse and have no effect on the resulting images as the training proceeds and FID get betters.
