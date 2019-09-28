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
│    │ xxx.mp4/0000.jpg
│    │ xxx.mp4/0001.jpg
│    │ ...
│    │ xxx.mp4/0002.jpg
│    │ ...
│
└───features/
│    │ frames-features.h5
│    │ videos-features.h5
│    │ ...
```

#### 2.1. Preprocessing
##### 2.1.1. Frame Extraction
Required files: videos in the folder: /path/to/data/videos/*.mp4.

Run the following command:
```bash
python videoprocess/frame_extraction.py --dataname svd
```
The extracted frames will be saved in the folder: /path/to/data/frames. We utilize 10 files: frames-[0-9].h5 to save frames. The total storage cost for frames is about 500G (516G on my device) when fps=1.
##### 2.1.2. Deep Features Extraction
Required files: frames-[0-9].h5 in the folder: /path/to/data/frames.

Run the following command:
```bash
CUDA_VISIBLE_DEVICES=1 python videoprocess/deepfeatures_extraction.py --dataname svd
```
The extracted deep features for each video will be saved in the file: /path/to/data/features/frames-features.h5. This file is about 246G when fps=1.
##### 2.1.3. Video Features Aggregations
Required files: frames-features.h5 in the folder: /path/to/data/features.
```bash
python videoprocess/videofeatures_extraction.py --dataname svd
```
The aggregated features for each will be saved in the file: /path/to/data/features/videos-features.h5. This file is about 8.8G when fps=1.
#### 2.1.4. Evaluation for Brute Force Search.
Required files: features in the folder: /path/to/data/features/videos-features.h5.

Run the following command:
```bash
python demos/bfs_demo.py --dataname svd
```
#### 2.2. Real-Value based Method
##### 2.2.1. CNNL
##### 2.2.2. CNNV
#### 2.3. Hashing based Method
##### 2.3.1. LSH
##### 2.3.2. ITQ
##### 2.3.3. IsoH
### 3. TODO list
+ [x] frame extraction
+ [] deep feature extraction
+ [] brute-force demo
+ [] CNNL/CNNV demo
+ [] LSH demo
+ [] ITQ demo
+ [] IsoH demo
+ [] Reranking demo

