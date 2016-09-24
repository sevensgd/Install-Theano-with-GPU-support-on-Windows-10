# Installing Theano with GPU support on Windows 10

## Visual Studio Community 2013
 - Download from https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx
 - add C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin to your PATH

## CUDA 7.5 (64-bit)
 - Download CUDA 7.5 (64-bit) from https://developer.nvidia.com/cuda-downloads
 - Install
 - Define a system environment variable CUDA_HOME C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
 - Add to the PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin
 - Add to the PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\libnvvp

## MinGW-w64 (5.3.0)
 - Download MinGW-w64, the newest one, there will be options during install https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/
 - Configuration: version 5.3.0, Architecture x86_64, Threads posix, Exception seh, Build version 0
 - Define the sysenv variable MINGW_HOME with the value C:\Program Files\mingw-w64\x86_64-5.3.0-posix-seh-rt_v4-rev0
 - Add %MINGW_HOME%\mingw64\bin to PATH
 
## Check if everything is ready in CMD
 - where gcc
 - where cl
 - where nvcc
 - where cudafe
 - where cudafe++
 
## ANACONDA
 - I got it already
 - conda install libpython
 
## GIT
 - Download and install https://git-scm.com/download/win
 
## Theano
- cd c:\TMP
- git clone https://github.com/Theano/Theano.git theano-0.8.2 --branch rel-0.8.2
- cd theano-0.8.2
- python setup.py install --record installed_files.txt
- check if it is there after install: conda list
 
## OpenBLAS 0.2.14
 - Downlad and install https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip/download
 - Unzip to C:\OpenBLAS-v0.2.14-Win64-int32
 - Define sysenv variable OPENBLAS_HOME
 - Add %OPENBLAS_HOME%\bin to PATH
 
## Switching between CPU and GPU mode
 
 - sysenv variable THEANO_FLAGS_CPU with the value:
 floatX=float32,device=cpu,lib.cnmem=0.8,blas.ldflags=-LC:/OpenBLAS-v0.2.14-Win64-int32/bin -lopenblas
 - sysenv variable THEANO_FLAGS_GPU with the value:
 floatX=float32,device=gpu,dnn.enabled=False,lib.cnmem=0.8,blas.ldflags=-LC:/OpenBLAS-v0.2.14-Win64-int32/bin -lopenblas
 - THEANO_FLAGS = %THEANO_FLAGS_GPU%
 
## Validating our OpenBLAS install

 ```python
 import numpy as np
 import time
 import theano
 
 print('blas.ldflags=', theano.config.blas.ldflags)
 
 A = np.random.rand(1000, 10000).astype(theano.config.floatX)
 B = np.random.rand(10000, 1000).astype(theano.config.floatX)
 np_start = time.time()
 AB = A.dot(B)
 np_end = time.time()
 X, Y = theano.tensor.matrices('XY')
 mf = theano.function([X, Y], X.dot(Y))
 t_start = time.time()
 tAB = mf(A, B)
 t_end = time.time()
 print("numpy time: %f[s], theano time: %f[s] (times should be close when run on CPU!)" % (
 np_end - np_start, t_end - t_start))
 print("Result difference: %f" % (np.abs(AB - tAB).max(), ))
 ```
 
## GPU test

```python 
 from theano import function, config, shared, sandbox
 import theano.tensor as T
 import numpy
 import time
 
 vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
 iters = 1000
 
 rng = numpy.random.RandomState(22)
 x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
 f = function([], T.exp(x))
 print(f.maker.fgraph.toposort())
 t0 = time.time()
 for i in range(iters):
     r = f()
 t1 = time.time()
 print("Looping %d times took %f seconds" % (iters, t1 - t0))
 print("Result is %s" % (r,))
 if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
     print('Used the cpu')
 else:
     print('Used the gpu')
```

## Installing Keras
 
 - cd c:\TMP
 - git clone https://github.com/fchollet/keras.git
 - python setup.py install --record installed_files.txt
 - check for results using conda list
 
## Validating Keras GPU
 
 - cd c:\TMP\keras-1.0.5\examples
 - python mnist_cnn.py
 
## Get a GPU monitor
  - Get and install https://www.techpowerup.com/downloads/2718/techpowerup-gpu-z-v0-8-9/mirrors
 
## cuDNN v5
 
 - https://developer.nvidia.com/rdp/cudnn-download
 - Choose the cuDNN Library for Windows10 dated May 12, 2016:
 - unzip cuda direcotry C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
 
 - THEANO_FLAGS_GPU_DNN with the following value:
 floatX=float32,device=gpu,optimizer_including=cudnn,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/OpenBLAS-v0.2.14-Win64-int32/bin -lopenblas
