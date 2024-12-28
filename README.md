# Pytorch Implementation for SGNetPose+

[Based on the original SGNet implementation](https://github.com/ChuhuaW/SGNet.pytorch)

Steps to get started

1. Download the dataset here
2. Run the following command to unpack the dataset, which will create a folder called jaadpie_pose
```
tar -xvf jaadpie_pose.tar.gz
```
3. Clone the SGNetPose repo
4. Modify the [custom_data_layer.py](lib/dataloaders/custom_data_layer.py) file at line 15 to refer to the directory where the jaadpie_pose directory is located
```python:lib/dataloaders/custom_data_layer.py [15]
```