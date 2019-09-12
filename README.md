# SVD-baselines
---

### 0. About the paper

This repo is the source code of the implementations of the baselines in the paper "SVD: A Large-Scale Short Video Dataset for Near-Duplicate Video Retrieval" publised on ICCV-2019. The authors are [Qing-Yuan Jiang](http://lamda.nju.edu.cn/jiangqy), Yi He, Gen Li, Jian Lin, Lei Li and Wu-Jun Li. If you have any questions about the source code, pls contact: jiangqy#lamda.nju.edu.cn or qyjiang#gmail.com

### 1. Running Environment
```bash
python 3
pytorch
tensorflow
```

### 2. Running Demos
#### 2.0. Preliminary
##### Directory to store some files
```
path/to/data
│
└───videos/
│    │ xxx.mp4
│    │ ...
│
└───frames/
│    │ frames-0.h5
│    │ ...
│    │ frames-9.h5
│    │ processed.pkl
│
└───features/
│    │ frames-features.h5
│    │ videos-features.h5
│    │ ...
```

#### 2.1. Preprocessing
##### 2.1.1. Frame Extraction
```bash
CUDA_VISIBLE_DEVICES=1 python videoprocess/deepfeatures_extraction.py.py --dataname svd
```
##### 2.1.2. Deep Features Extraction
##### 2.1.3. Video Features Aggregations
#### 2.2. Real-Value based Method
##### 2.2.1. CNNL
##### 2.2.2. CNNV
#### 2.3. Hashing based Method
##### 2.3.1. LSH
##### 2.3.2. ITQ
##### 2.3.3. IsoH

