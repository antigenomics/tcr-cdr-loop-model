# Loop coordinates prediction model

Notebooks:

- `extract.ipynb` - 

- `eda.ipynb` - 

- `aa_predict.ipynb` - 

- `aa_predict_clust.ipynb` - 

- `cluster.ipynb` - 

- `coord_predict.ipynb` - 

Folders:

- `wisp` - folder with code for model training.

- `data` - folder with datasets.

- `loss` - visualisation of training losses.

- `pred` - visualisation of predicted coordinates.

- `plots` - various plots.


## 1. Preprocessing

Data is highly variable due to different structures and stochasticity in the optimization procedure (length - 11, X and Y coordinates, distribution of differences among the max and min values per position):

<img src="plots/maxmin_len11_x.png">

<img src="plots/maxmin_len11_y.png">

There are no critical differences for CDR sequences, therefore for each sequence we chose the mean coordinates. However, canonical and putative sequences are much more diverse and more sophisticated procedure is needed. For each unique sequence we choose each other kmer with differences in MSE and minimal MSE (per sequence) lower than 0.005:

<img src="plots/maxmin_len11_x_proc.png">

<img src="plots/maxmin_len11_y_proc.png">

Then we choose the mean coordinate values (like in CDRs) per sequence.

Numbers of the sequences / unique sequences per length per data type:

```
CDR
L all uniq
10 1  1
11 21 14
12 32 16
13 69 37
14 46 31
15 42 25
16 32 11

Canonical
L all uniq
10 176 60
11 910 171
12 452 97
13 395 84
14 167 24
15 109 17
16 80  4

Putative
L all uniq
10 7219   1648
11 179175 23598
12 146165 17408
13 74735  10167
14 7177   1463
15 7357   1254
16 2697   481
```


## 2. Features


### 1.1. Naive approach

Surround sequences by null symbols.


### 1.2. "Omega loops"

First and last amino acids depends on each other other. It is better than the naive approach:
<img src="loss/loss_x_dense_4_4_comparison_clust_and_onehot_1200it.png" width="80%">


### 1.3. Positional / length-positional

Add position or position/length for each amino acid to each dense layer. It helps:

<img src="loss/comparison/loss_x_dense_comparison_1500it.png" width="80%">

Interstinegly enough, if we train the model on CDRs and predict putative ones than the pos-len modification helps a lot:

<img src="loss/loss_dense_changeCDRs_1500it.png" width="80%">


### 3.4. Delta models - difference between neighbour coordinates


## 3. Models

### 3.1. Convolutional

### 3.2. Recurrent

### 3.3. Convolutional-recurrent

--

## 1. Coordinates and models

**window** - sliding window, predict the coordinate in the centre of the window.

**directional / bidirectional** - recurrent neural networks.

**straight** - predict each coordinate.

**difference** - predict differences between neighbour coordinates

**window + straight** - predict each coordinate.

### 1.1. Naive approach

Surround sequences by null symbols.

### 1.2. "Omega loops"

First and last amino acids depends on each other other. It is better than the naive approach:
<img src="loss/loss_x_dense_4_4_comparison_clust_and_onehot_1200it.png" width="80%">

### 1.3. Positional / length-positional

Add position or position/length for each amino acid to each dense layer. It helps:

<img src="loss/comparison/loss_x_dense_comparison_1500it.png" width="80%">

Interstinegly enough, if we train the model on CDRs and predict putative ones than the pos-len modification helps a lot:

<img src="loss/loss_dense_changeCDRs_1500it.png" width="80%">

### 1.4. Recurrent models

GRU models, 1500 iterations:
<img src="loss/comparison/loss_x_gru_comparison_1500it.png" width="80%">

### 1.5. Convolutional models

GRU models, 1500 iterations:
<img src="loss/comparison/loss_x_cnnpos_comparison.png" width="80%">

### 1.6. Difference among coordinates (left-to-right / right-to-left directions)

#### Naive prediction

Due to the iterative nature the error is aggreageted through the sequence.

<img src="pred/pred_x_diff_left_coords.png" width="80%">

<img src="pred/pred_x_diff_right_coords.png" width="80%">


#### 1.6.2. Ensembling

Ensemble the predicted coordinates with `Bayesian Ridge`. Cross validation score (negative MSE): 

```
array([-0.0673693 , -0.1109571 , -0.39486038, -0.09391625, -0.02850241,
       -0.13117736, -0.10696883, -0.18005811, -0.13272101, -0.20850138])
```

<img src="pred/pred_x_ensembling.png">


## 2. Amino acid transformations

**one-hot** - code each amino acid as a 20-dimensional vector with 1 at the index of the corresponding amino acid.

**kidera** - worked worse than one-hot.

**embeddings** - ??? (infer embeddings on the overall dataю)

**two-hot** - one-hot vector + one-hot vector for target amino acid.


## 3. Feature preprocessing

### 3.1. Scale the data

Scale the data to [0,1]. Tested on models with [128,64] dense layers, 4-4 window size, 2000 iterations.
- per column - scale each column independently:
```
MSE:
no scale:
    cdr - 1.61775   ***
    can - 0.0993797
MinMax:
    cdr - 2.43089   *
    can - 0.0978544
MaxAbs:
    cdr - 1.95208   **
    can - 0.0967865
```
- overall - scale the overall matrix:
```
no scale:
    cdr - 1.61775   ***
    can - 0.0993797
MinMax:
    cdr - 2.27693   *
    can - 0.105339
MaxAbs:
    cdr - 1.96837   **
    can - 0.0973526
```

### 3.2. Pre-clustering

Assign weights accroding to the clusters' sizes. Cluster weight = `ln(cluster size) / ln(minimal cluster size)`. Clustering helps:

<img src="loss/loss_x_dense_onehot_noscale_clust5.png" width="90%">

However prediction with clustering is a little bit off due to the small weights on big clusters:

<img src="pred/pred_strange_with_clustering.png" width="110%">


### 3.3. Pre-clustering with fading weights on clusters

???
???

<img src="loss/loss_x_dense_onehot_noscale_clust5_fade.png" width="90%">


## 4. Learning

### 4.1. Straightforward learning

<img src="loss/loss_x_densepos_4_4_comparison_clust_and_onehot_1200it.png" width="80%">


### 4.2. Change learning rate

Factor 0.3, patience 3
<img src="loss/loss_x_densepos_4_4_comparison_clust_and_onehot_changelr_factor03_patience3_1200it.png" width="80%">

Factor 0.1, patience 3
<img src="loss/loss_x_densepos_4_4_comparison_clust_and_onehot_changelr_factor01_patience3_1200it.png" width="80%">

Factor 0.1, patience 6
<img src="loss/loss_x_densepos_4_4_comparison_clust_and_onehot_changelr_factor01_patience6_1200it.png" width="80%">


### 4.3. Add putative sequences to batches

??? (to add noise)


## 5. Post-analysis

### 5.1. Ensembling

???