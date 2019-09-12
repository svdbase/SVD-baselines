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
Required files: videos in the folder: /path/to/data/videos/.

Run the following command:
```bash
python videoprocess/frame_extraction.py --dataname svd
```
The extracted frames will be saved in the folder: /path/to/data/frames. We utilize 10 files: frames-[0-9].h5 to save frames. The total storage cost for frames is about xxxG when fps=1.
##### 2.1.2. Deep Features Extraction
Required files: frames-[0-9].h5 in the folder: /path/to/data/frames.

Run the following command:
```bash
CUDA_VISIBLE_DEVICES=1 python videoprocess/deepfeatures_extraction.py.py --dataname svd
```
The extracted deep features for each video will be saved in the file: /path/to/data/features/frames-features.h5. This file is about xxxG when fps=1.
##### 2.1.3. Video Features Aggregations
Required files: frames-features.h5 in the folder: /path/to/data/features.
```bash
python videoprocess/videofeatures_extraction.py --dataname svd
```
The aggregated features for each will be saved in the file: /path/to/data/features/videos-features.h5. This file is about XXXG when fps=1.
#### 2.1.4. Evaluation for Brute Force Search.

#### 2.2. Real-Value based Method
##### 2.2.1. CNNL
##### 2.2.2. CNNV
#### 2.3. Hashing based Method
##### 2.3.1. LSH
##### 2.3.2. ITQ
##### 2.3.3. IsoH

