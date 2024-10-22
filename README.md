# RR-PointHop

## Packages

The code has been developed and tested in Python 3.6. The following packages need to be installed.

```
h5py
numpy
scipy
sklearn
open3d
```

## Training

Train the model on all 40 classes of ModelNet40 dataset

```
python train.py --first_20 False
```

Train the model on first 20 classes of ModelNet40 dataset

```
python train.py --first_20 True
```

User can specify other parameters like number of points in each hop, neighborhood size and energy threshold, else default parameters will be used.

## Registration 

```
python test.py 
```

