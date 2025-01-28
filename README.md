# Pytorch Implementation for SGNetPose+

[Based on the original SGNet implementation](https://github.com/ChuhuaW/SGNet.pytorch)

Steps to get started

1. Download the dataset [here](https://drive.google.com/drive/folders/1w6V40nL_4Keg4H9SDjoeXmwJ5KECuFVG)
2. Run the following command to unpack the dataset, which will create a folder called jaadpie_pose
```
tar -xvf jaadpie_pose.tar.gz
```
3. Clone the SGNetPose repo
4. Modify the [custom_data_layer.py](lib/dataloaders/custom_data_layer.py) file to refer to the directory where the jaadpie_pose directory is located

https://github.com/Aghiya/SGNetPose/blob/3e2f4914b387e8d6a1ccabdf5a445cb4fddf8ede/lib/dataloaders/custom_data_layer.py#L14

5. Create a conda environment using the provided YAML file, the environment name in the file is sgnetpose
```
conda env create -f sgnetpose.yml
```
6. The [1_jaad.sh](1_jaad.sh) and [2_pie.sh](2_pie.sh) scripts are set up to make the process of running the model easy. They create timestamped logs and start the process in screens so that disconnecting from the server or losing connection doesn't kill the process. You can also modify hyperparameter values like the batch size, number of epochs, and the seed. For your part, you will need to modify the conda type if you aren't using miniconda3. Also, verify the following directory paths.

https://github.com/Aghiya/SGNetPose/blob/88e4d08dfe31b223382e9d6ed9a56598f11e1a5e/1_jaad.sh#L9-L10

That's it! You should be able to run the model as you wish now.
