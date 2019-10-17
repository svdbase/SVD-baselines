# SVD-baselines
---

### 0. About the paper

This repo is the source code of the implementations of the baselines in the paper "SVD: A Large-Scale Short Video Dataset for Near-Duplicate Video Retrieval" publised on ICCV-2019. The authors are [Qing-Yuan Jiang](http://lamda.nju.edu.cn/jiangqy), Yi He, Gen Li, Jian Lin, Lei Li and Wu-Jun Li. If you have any questions about the source code, pls contact: jiangqy#lamda.nju.edu.cn or qyjiang#gmail.com

### 1. Running Environment
```bash
python 3
pytorch
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
│    │ xxxx.mp4/0000.jpg
│    │ xxxx.mp4/0001.jpg
│    │ ...
│    │ xxxx.mp4/xxxx.jpg
│    │ ...
│    │ xxxy.mp4/0000.jpg
│    │ xxxy.mp4/0001.jpg
│    │ ...
│    │ xxxy.mp4/xxxx.jpg
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
The extracted frames will be saved in the folder: /path/to/data/frames/. The total storage cost for frames is about 400G (358G on my device) when fps=1.
##### 2.1.2. Deep Features Extraction
Required files: frames/xxx.mp4/xxxx.jpg in the folder: /path/to/data/frames.

Run the following command:
```bash
CUDA_VISIBLE_DEVICES=1 python videoprocess/deepfeatures_extraction.py --dataname svd
```
The extracted deep features for each video will be saved in the file: /path/to/data/features/frames-features.h5. This file is about 153G when fps=1.
##### 2.1.3. Video Features Aggregations
Required files: frames-features.h5 in the folder: /path/to/data/features.
```bash
python videoprocess/videofeatures_extraction.py --dataname svd
```
The aggregated features for each will be stored in the file: /path/to/data/features/videos-features.h5. This file is about 8.8G when fps=1.
#### 2.1.4. Evaluation for Brute Force Search.
Required files: features in the folder: /path/to/data/features/videos-features.h5.

Run the following command:
```bash
python demos/bfs_demo.py --dataname svd
```

The map is: 0.7537
#### 2.2. Hashing based Method
Required file: features in the folder: /path/to/data/features/videos-features.h5.
##### 2.2.1. LSH
Run the following command:
```bash
python demos/lsh_demo.py --dataname svd --approach lsh --bit 16
```

The map is: 0.0370
##### 2.2.2. ITQ
Run the following command:
```bash
python demos/itq_demo.py --dataname svd --approach itq --bit 16
```

The map@16bits is: 0.0560
##### 2.2.3. IsoH
Run the following command:
```bash
python demos/isoh_demo.py --dataname svd --approach isoh --bit 16
```

The map@16bits is: 0.0562

#### 2.3. Real-Value based Method
##### 2.3.1. CNNV
+ Step 1: sampling frames for clustering:

Required files: frames-features.h5 in the folder: /path/to/data/features.

```bash
python videoprocess/cnnlv_keyframe_sampling.py --dataname svd --approach cfs
```

Sampled frames are stored at: /path/to/data/features/cnnlv-sampling-features.h5

+ Step 2: clustering

Required files: cnnlv-sampling-features.h5 in the folder: /path/to/data/features.

```bash
python videoprocess/cnnv_clustering.py --dataname svd --approach cnnvcluster
```

The learned centers are stored at: /path/to/data/features/cnnv-centers.h5

+ Step 3: generating cnnv features

Required files: cnnv-centers in the folder: /path/to/data/features/cnnv-centers.h5

```bash
python videoprocess/cnnv_feature_aggregation.py --dataname svd --approach cnnvfa
```

The learned features are stored at: /path/to/data/features/cnnv-agg-features.h5

+ Step 4: evaluating cnnv

Required files: cnnv-aggregated features in the folder: /path/to/data/features/cnnv-agg-features.h5

```bash
python demos/cnnv_demo.py --dataname svd --approach cnnv
```

The map is: 0.1895
### 3. TODO list
+ [x] frame extraction
+ [x] deep feature extraction
+ [x] brute-force demo
+ [x] LSH demo
+ [x] ITQ demo
+ [x] IsoH demo
+ [x] CNNV demo

