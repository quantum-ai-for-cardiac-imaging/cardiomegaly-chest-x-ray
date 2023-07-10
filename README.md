# Hybrid Classical-Quantum Transfer Learning for Cardiomegaly Detection on Chest X-Rays

![](/main_diagram.png)

## Open Access Article

J. Imaging 2023, 9(7), 128; https://doi.org/10.3390/jimaging9070128

## Dataset
CSV files are available in the subfolders of the chexpert-corrected folder, organized to comply with the codes.

These subfolders must be populated by the corresponding images to download from:
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
after login and agreeing to the Stanford University Dataset Research Use Agreement.

## Code
Author: [@poig](https://github.com/poig)

Notebook Version:

[init_freezer.ipynb](/init_freezer.ipynb)

[no freezer.ipynb](/no%20freezer.ipynb)

[init_freezer - P6qubit.ipynb](/init_freezer%20-%20P6qubit.ipynb)

[init_freezer - P8qubit.ipynb](/init_freezer%20-%20P8qubit.ipynb)

[init_freezer - P10qubit.ipynb](/init_freezer%20-%20P10qubit.ipynb)

[saliency-maps-test-set-classical-classical-model.ipynb](/saliency-maps-test-set-classical-classical-model.ipynb)

[ten-fold-cross-validation-cc-densenet-121.ipynb](/ten-fold-cross-validation-cc-densenet-121.ipynb)

Command Prompt Version:

`command prompt/trainer.py` : the main program here you will execute in command prompt

`command prompt/requirements.txt` : requirement package install with `conda create --name <env> --file <this file>`

`command prompt/command` : example how to start training, require modify address before execute

## Saliency maps

Saliency maps obtained using GradCAM++ for the test dataset are collected in folder [gradcam_final](/gradcam_final/).
  - Subfolder gcnn is for the classical-classical model.
  - Subfolder gpnn is for the PennyLane-based classical-quantum model.
  - Subfolder gqnn is for the Qiskit-based classical-quantum model.

The notebooks that generated these collections are:

[saliency-maps-test-set-classical-model.ipynb](/saliency-maps-test-set-classical-model.ipynb)

[saliency-maps-test-set-6-qubit-qiskit-model.ipynb](/saliency-maps-test-set-6-qubit-qiskit-model.ipynb)

[saliency-maps-test-set-6-qubit-pennylane-model.ipynb](/saliency-maps-test-set-6-qubit-pennylane-model.ipynb)
