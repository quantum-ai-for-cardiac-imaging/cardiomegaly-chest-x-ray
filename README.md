# Hybrid Classical-Quantum Transfer Learning for Cardiomegaly Detection on Chest X-Rays

![](/main_diagram.png)

## dataset
CSV files are available in the subfolders of the chexpert-corrected folder, organized to comply with the codes.

These subfolders must be populated by the corresponding images to download from:
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
after login and agreeing to the Standford University Dataset Research Use Agreement.

## Code

Notebook Version:

[week5_realdevice_v2(init_freezer).ipynb](https://github.com/quantum-ai-for-cardiac-imaging/cardiomegaly-chest-x-ray/blob/main/week5_realdevice_v2(init_freezer).ipynb)

[week5_realdevice_v2(no freezer).ipynb](https://github.com/quantum-ai-for-cardiac-imaging/cardiomegaly-chest-x-ray/blob/main/week5_realdevice_v2(no%20freezer).ipynb)

[week5_realdevice_v2(init_freezer) - 6qubit.ipynb](https://github.com/quantum-ai-for-cardiac-imaging/cardiomegaly-chest-x-ray/blob/main/week5_realdevice_v2(init_freezer)%20-%206qubit.ipynb)

[week5_realdevice_v2(init_freezer) - 8qubit.ipynb](https://github.com/quantum-ai-for-cardiac-imaging/cardiomegaly-chest-x-ray/blob/main/week5_realdevice_v2(init_freezer)%20-%208qubit.ipynb)

[week5_realdevice_v2(init_freezer) - 10qubit.ipynb](https://github.com/quantum-ai-for-cardiac-imaging/cardiomegaly-chest-x-ray/blob/main/week5_realdevice_v2(init_freezer)%20-%2010qubit.ipynb)

Command Prompt Version:

`command prompt/trainer.py` : the main program here you will execute in command prompt

`command prompt/requirements.txt` : requirement package install with `conda create --name <env> --file <this file>`

`command prompt/command` : example how to start training, require modify address before execute
