<p>
    <h1 align="center">
        Real-time Dynamic Hand Gesture Recognition
    </h1>
</p>
<br>

## Models

### DD-net

We use the [DHG-14/28](http://www-rech.telecom-lille.fr/DHGdataset/) dataset,
specifically the `skeleton_world.txt` file of this dataset for DDnet training.
Download the specific file using the link below. We are not the creators of this
dataset and hence would like to give full credits to the original authors.

[Google Drive for skeleton_world.txt](https://drive.google.com/file/d/1lEKmgiMVcEIOyB4ABItfjZ4lwrPVsNX6/view?usp=sharing)

### SRN

Download pretrained model and put it in as `./srn/checkpoint/offset_000_240.pth/`.

[Google Drive for pretrained MSRA model](https://drive.google.com/drive/folders/1QG6F9aD4t-LLupoguWVpBm-fUyGPNRl0)

## Developers

- Manan Gandhi ([manangandhi-06](https://github.com/manangandhi-06))
- Preet Shah ([preet4737](https://github.com/preet4737))
- Vikrant Gajria ([vixrant](https://github.com/vixrant))

## References

Yang, Fan, Yang Wu, Sakriani Sakti, and Satoshi Nakamura. "Make Skeleton-based Action Recognition Model Smaller, Faster and Better". 

Pengfei Ren, Haifeng Sun, Qi Qi, Jingyu Wang, Weiting Huang. "SRN: Stacked Regression Network for Real-time 3D Hand Pose Estimation". 

