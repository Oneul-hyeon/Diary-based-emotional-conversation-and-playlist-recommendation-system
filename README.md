# Epilogue Of The Day
This project was carried out in Colab.

If you want to proceed with this project as a whole, proceed with the preprocessing process in the Dataset folder first.

In this project, seven emotions are classified through text.

- 행복
- 보통
- 슬픔
- 분노
- 놀람
- 불쾌함
- 두려움

## Modeling
The entire modeling process was carried out in the Modeling.ipynb file.

For modeling to proceed, the following files must exist in the root folder.

- train_data.pkl
- test_data.pkl

This file is a file that has undergone preprocessing in the Dataset folder.

### Environment Setting
To run Modeling.ipynb, library and package installation process is required.

#### When using in colab
The code for installing libraries and packages is at the top of Modeling.ipynb.

You can proceed further by running the code.

#### When using a virtual environment
Currently, Colab's Python is version 3.10.

So follow the below method in anaconda terminal.

Clone this repository:

```
git clone https://github.com/toy-f-rebellion/toy_ai.git
cd {location of that folder}
```

Install Python and other dependencies:

```
conda create -n {env_name} python=3.10
conda activate {env_name}
pip install -r requirements.txt
```

This file is the overall library and package installation file of the AI part of this project.

This process allows you to run the Modeling.ipynb file.

Also, in this case, delete the library and package installation block at the top of the file.

Since this file was created in Colab, it is necessary to reset the file path to load or save in order to proceed in the virtual environment.

## Pretrained Model
You can obtain the derived Pretrained Model through the Modeling.ipynb file through the link below.

- https://drive.google.com/file/d/1X3eKRDdwLXy1y1RhgBrc4F4MDmfWLX8W/view?usp=sharing

After downloading the file, save the model.pth file in the root folder.

## Caution
The current model is using the GPU.

Therefore, if you want to use the CPU, you need to modify the device configuration part of pytorch.

Also, when you want to get the output value through evaluate.py, you need to modify the path where the model.pth file is saved.

Following these two steps will give you the desired result.