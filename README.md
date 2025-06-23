# AppleGrowthVision

Large-scale heterogeneous apple tree data for BBCH classification, Apple Detection, and 3D reconstruction of orchards with sparse data.

[Project Page](https://fraunhoferhhi.github.io/AppleGrowthVision)

## BBCH classification

1. Download the data
2. Run `prepare_bbch.py` to copy and rename the data for bbch classification. The data can then be used via the    `torchvision.datasets.ImageFolder` class.

```bash
python prepare_bbch.py -p <path to directory "AppleGrowthVision"> -o <path to directory of final dataset>
```

## Stereo Reconstruction

The script `calib_conversion.py` provides a function `convert_xml_to_cameras(path)` to retrieve the calibration parameters for a given capture date. It returns the parameters in openCV format for each camera (L, R).

## Dataset

The dataset consists of two subsets that have different modalities. `Brandenburg` was captured using a calibrated stereo setup. `Saxony` was captured by a smartphone. The data can be found [here](https://fraunhoferhhi.github.io/AppleGrowthVision).

### Brandenburg

```
- calib/
  - <capture date>.xml
- data/
  - <capture date>/
    - L_*.jpg
    - R_*.jpg
  - annotations.json
```

The brandenburg data has a directory `calib` which contains all calibration parameters per capture date. It might be converted using `calib_conv.py`.

The directory `data` contains images per capture date named as `<camera>_<image pair>.jpg`. Additionally, bounding boxes for apple detection are provided in the `annotations.json`.

### Saxony

```
- 2024_counted_trees
  - counted_trees_images/
    - <row id>_<tree id>/
      - *.jpg
  - annotations.json
  - 2024_counted_trees.json
- <year>/
  - <date>_<time>_pillnitz_<growthstage>_tree_<row id>-<tree id>_<picture id>.jpg
- annotations.json
```

The saxony data contains directories for each year. The images inside the year directory are named as `<date>_<time>_pillnitz_<growthstage>_<row id>-<tree id>_<picture id>.jpg`.

Valid growthstages are:

- apple blossom
- apple small fruit
- apple middle fruit
- apple fruit

The directory `2024_counted_trees` contains human annotations for 10 trees with 6 orbital images and annotated bounding boxes:

| row-index | tree-index | tree-id |    apple count    |
| :-------: | :--------: | :-----: | :----------------: |
|    36    |     2     |  36.2  |         23         |
|    36    |     5     |  36.5  |         1         |
|    37    |     2     |  37.2  |         4         |
|    44    |     65     |  44.65  |         28         |
|    45    |     9     |  45.9  |         14         |
|    45    |     49     |  45.49  |         32         |
|    46    |     4     |  46.4  | 11 + 1 bad quality |
|    46    |     63     |  46.63  | 24 + 2 bad quality |
|    47    |     8     |  47.8  |         15         |
|    47    |     12     |  47.12  | 28 + 4 bad quality |

## Citation

If the provided data is used or any of the provided scripts, please cite:

```
@InProceedings{von_Hirschhausen_2025_CVPR,
    author    = {von Hirschhausen, Laura-Sophia and Magnusson, Jannes S. and Kovalenko, Mykyta and Boye, Fredrik and Rawat, Tanay and Eisert, Peter and Hilsmann, Anna and Pretzsch, Sebastian and Bosse, Sebastian},
    title     = {AppleGrowthVision: A large-scale stereo dataset for phenological analysis, fruit detection, and 3D reconstruction in apple orchards},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {5443-5450}
}
```
