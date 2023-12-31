# Sky_Seg_Trainer

## Overview
Sky_Seg_Trainer is an application designed for segmenting images to determine the line-of-sight (LOS) for satellites in the sky. The tool is capable of distinguishing satellites that are visible above the horizon using advanced image segmentation techniques implemented via the DeepLabV3+ model.

<p align="center">
  <img width="768" alt="image" src="https://github.com/SeanBaek111/Sky_Seg_Trainer/assets/33170173/b8a7b354-25c5-4b27-a999-a0ed31bbc63b">
</p>
## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/YourUsername/Sky_Seg_Trainer.git
cd Sky_Seg_Trainer
```

## Usage
To start using the Sky_Seg_Trainer, run the RunTrainer.ipynb Jupyter notebook:

```bash
jupyter notebook RunTrainer.ipynb
```
Ensure to specify the dataset paths in the notebook:
```bash
DATASET_PATH = 'path_dataset'
DOWNLOAD_PATH = 'path_to_download'
```
Use the downloader.download_ade function within the notebook to fetch the dataset:

```bash
downloader.download_ade()
```
# Image Segmentation for Sky-Based Satellite LOS Determination
## Sky Classification:
The application integrates transformed satellite coordinates into an augmented reality view, allowing users to visually identify GNSS satellite locations through their iPhone display.

## Implementation:
The implementation leverages the DeepLabV3+ model with a ResNet50 backbone, enhanced by transfer learning with weights pre-trained on ImageNet. This approach facilitates precise segmentation necessary for reliable satellite visibility analysis.

## Contributing
Contributions to Sky_Seg_Trainer are welcome! Please feel free to fork the repository, make your changes, and create a pull request to contribute.

## License
This project is available under the MIT License. See the [LICENSE](https://github.com/SeanBaek111/Sky_Seg_Trainer/blob/main/LICENSE) file for more details.

Questions and Support
For questions and support, please open an issue in the GitHub repository.
