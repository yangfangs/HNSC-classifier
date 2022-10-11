# **HNSC-classifier: an accuracy tool for head and neck cancer detection in digitized whole-slide histology using deep learning**

an accuracy tools for head and neck cancer detection and stage inferred in digitized whole-slide histology using deep learning

# The HNSC-classifier scheme and Deep learning framework:

![Workflow](https://github.com/yangfangs/HNSC-classifier/blob/main/scheme/scheme.png)


### options

| option | Description                                                                                                                          |
|:-------|:-------------------------------------------------------------------------------------------------------------------------------------|
| -i     | Path to a whole slide image                                                                                                          |
| -o     | Name of the output file directory [default: `output/`]"                                                                              |
| -p     | The pixel width and height for tiles                                                                                                 |
| -l     | Extract tiles form resolution of level                                                                                               |
| -c     | The deep model path of cancer/normal classification                                                                                  |
| -s     | The deep model path of stage classification                                                                                          |
| -t     | The deep model path of T classification (TNM Staging System)                                                                         |
| -n     | The deep model path of N classification (TNM Staging System)                                                                         |
| -m     | The deep model path of M classification (TNM Staging System)                                                                         |

### Dependents

```angular2html
pandas==1.4.3
pillow==8.4.0
matplotlib==3.5.2
scipy==1.8.0
numpy==1.22.3
openslide-python
fastai==2.7.9
histolab==0.5.1
```


### Installation:
1. install system dependency:

HNSC-classifier has one system-wide dependency: `OpenSlide`.

You should first download and install it from https://openslide.org/download/ according to your operating system.

2. install HNSC-classifier

```angular2html
$pip install HNSC-classifier
```


### Usage:

```angular2html
$ HNSC-calssifier - i TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs -c learn.pkl
```

# Example (test in linux OS: Ubuntu 20.4, python 3.9)

## Download test data
The test Whole slide image download form TCGA [TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs](https://bit.ly/3CqLyJt).

## Download deep learning model
| DP model        | tarin tiles                                                  | Description                                                   |
|:----------------|:-------------------------------------------------------------|:--------------------------------------------------------------|
| [learn.pkl](https://bit.ly/3RKzVTy)   | 1,392,135                                                    | The deep learn model for detected tumor/normal                |
| [learn_S.pkl](https://bit.ly/3EtE7nx) | 1,428,765                                                    | The deep learn model for classified stage                     |
| [learn_M.pkl](https://bit.ly/3CqM0r9) | 1,428,765                                                    | The deep model for classified stage M (TNM Staging System)    |
| [learn_N.pkl](https://bit.ly/3SPPbjq) | 1,428,765                                                    | The deep model for classified stage N (TNM Staging System)    |
| [learn_T.pkl](https://bit.ly/3EAF25J) | 1,428,765                                                    | The deep model for classified stage T (TNM Staging System)    |
 
> If you can not clink the hyperlink to obtain test data and DP model, you can download test data from `ftp://23.105.208.65`

## Run HNSC-classifier in virtualenv
1. install virtualenv

```angular2html
$ pip install virtualenv
```

2. Create virtual environment 
```angular2html
$ virtualenv ven
```
3. Activate environment
```angular2html
$ source ven/bin/activate
```
4. install HNSC-classifier

```angular2html
$pip install HNSC-classifier
```
5. validate installation

```angular2html
$HNSC-classifier -h
```

## HNSC-classifier for cancer detected.

```angular2html
$ HNSC-calssifier - i TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs -c learn.pkl
```
## HNSC-classifier for stage detected.

```angular2html
$ HNSC-calssifier - i TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs -s learn_S.pkl
```
                                                        
## HNSC-classifier for TNM Staging System detected.

```angular2html
$ HNSC-calssifier - i TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs -t learn_T.pkl -m learn_M.pkl -n learn_N.pkl
```
## Output
```
Extract_tiles/  
                tile_0_level0_1499-7466-1723-7690.png
                tile_1_level0_1499-7690-1723-7914.png
                tile_2_level0_1499-8810-1723-9034.png
                tile_3_level0_1499-9034-1723-9258.png
                ...
cancer_heatmap.png
stage_heatmap.png
TNM_system_M_heatmap.png
TNM_system_N_heatmap.png
TNM_system_T_heatmap.png
summary.png
summary.csv

```
* **Extract_tiles**: the tiles extract from WSI. 
* **cancer_heatmap.png**: cancer detected result.
* **stage_heatmap.png**: stage detected result.
* **TNM_system_M_heatmap.png**: TNM stage system (M) detected result.
* **TNM_system_N_heatmap.png**: TNM stage system (N) detected result.
* **TNM_system_T_heatmap.png**: TNM stage system (T) detected result.
* **summary.png**: the summary of extracted and predicted tiles info.
* **summary.csv**: the summary of extracted and predicted tiles info.
