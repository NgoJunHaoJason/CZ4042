# CZ4042 project

Nanyang Technological University  
School of Computer Science and Engineering

Academic Year 2020-2021 Semester 1

source code for CZ4042 Neural Networks and Deep Learning project

---

notebooks written by me:

- `3_modelling/model_evaluation/hassner_cnn.ipynb`
- `3_modelling/model_evaluation/mobilenetv2.ipynb`
- `7_hyperparameter_tuning/hyperparams_tuning_batch128.ipynb`
- `7_hyperparameter_tuning/hyperparams_tuning_batch64.ipynb`
- `7_hyperparameter_tuning/hyperparams_tuning_L2.ipynb`
- `7_hyperparameter_tuning/hyperparams_tuning_dropout.ipynb`

---
Finding Files
---
1. Open File Directory.xlsx for the description of the files

---
Running Notebooks
---
There are 2 ways of running the notebooks. 

-------------Method 1-------------
1. Create a free account on https://www.wandb.com/
2. Install the wandb library using pip (pip install wandb)
3. Run the cell in the notebook
4. When prompted for the API code, paste the api code and press Enter to login
5. Ensure that the directory path to aligned_gender.txt is correct when loading the dataframe using pd.readcsv()
6. Ensure that the image path is correctly added to the dataframe

-------------Method 2-------------
Delete all the wandb code
1. Delete wandb.init() and config = wandb.config
2. Remove the defaults dictionary and use variables instead for the hyperparameters
3. Remove the [WandbCallback()] parameter under model.fit()
4. Replace all 'config.' to '' as they are not needed unless you're using wandb
5. Ensure that the directory path to aligned_gender.txt is correct when loading the dataframe using pd.readcsv()
6. Ensure that the image path is correctly added to the dataframe

* Hyperparameter sweeps are exclusive to wandb. Method 2 will not be able to run sweeps. 
* Most notebooks are done in Google Colab and downloaded on Google Drive. 
In the event that you are unable to run it, do add .ipynb to the file name
