you will need to install `nvidia-cudnn`, it depends on your system version, otherwise you will get this kernel error:  
```warn 21:36:08.336: StdErr from Kernel Process Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory```  
this will help you https://github.com/tensorflow/tensorflow/issues/45200  
Note: it might take long time to download and install, 

you also need pip install `qiskit-aer-gpu` and `cuquantum`

for window, you will need to install cuda toolkit, you can check which version to download here, https://pytorch.org/get-started/locally/
