# Hybrid Classical-Quantum Transfer Learning for Cardiomegaly Detection on Chest X-Rays

![](/main_diagram.png)

## dataset
CSV files are available in the subfolders of the chexpert-corrected folder, organized to comply with the codes.

These subfolders must be populated by the corresponding images to download from:
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
after login and agreeing to the Standford University Dataset Research Use Agreement.

## Code

Notebook Version:

[init_freezer.ipynb](/init_freezer.ipynb)

[no freezer.ipynb](/no%20freezer.ipynb)

[init_freezer - P6qubit.ipynb](/init_freezer%20-%20P6qubit.ipynb)

[init_freezer - P8qubit.ipynb](/init_freezer%20-%20P8qubit.ipynb)

[init_freezer - P10qubit.ipynb](/init_freezer%20-%20P10qubit.ipynb)

Command Prompt Version:

`command prompt/trainer.py` : the main program here you will execute in command prompt

`command prompt/requirements.txt` : requirement package install with `conda create --name <env> --file <this file>`

`command prompt/command` : example how to start training, require modify address before execute
