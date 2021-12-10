# Simple Explanation
Here's simple code that I used for this challenge.

image_extarct.py : It's for image preprocess. Taking only human images from  raw datasets, deleting masked head part that is not needed and flipping images and mask while left and right body segmentations to be changed are all the things.

utils.py : Here's utility functions or classes that I used. Dataset class, augmentation functions.

model.py : My basic model architecture codes. I tried to write down codes from scratch.

train.py : Codes for training.

mask_prediction.py : Codes for mask prediction using trained model.


## More details
As the challenge is about semantic segmentation problem, I tried to search some papers. And I found it!

- The surprising impact of mask-head architecture on novel class segmentation(https://openaccess.thecvf.com/content/ICCV2021/papers/Birodkar_The_Surprising_Impact_of_Mask-Head_Architecture_on_Novel_Class_Segmentation_ICCV_2021_paper.pdf)
    - The author proposed novel idea(at least for me), but it was not my interest. It found Stacked Hour Glass Model is really powerful segmentation model when the model becomes deeper. So from this point, I tried to write down own my codes.

- Stacked Hourglass Networks for Human Pose Estimation(https://arxiv.org/pdf/1603.06937.pdf)
    - It's the original paper. I followed their guide but as there's no public code, some changes may be applied.
