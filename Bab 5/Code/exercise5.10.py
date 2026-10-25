# Exercise 5.10 VGG16 as the base model for brain image classification
# The RSNA Intracranial Hemorrhage dataset in the book is 427 GB of DICOM CT scans, far too big
# to download here. The same exercise is done on the book's other brain dataset (section 5.4.5):
# Kaggle "brain-mri-images-for-brain-tumor-detection" (98 no tumor / 155 tumor).
# The RSNA Kaggle notebook uses InceptionV3, so VGG16 is compared with InceptionV3.
from transfer_utils import kaggle_download, find_class_root, compare_models

path = kaggle_download("navoneel/brain-mri-images-for-brain-tumor-detection")
data_dir = find_class_root(path)

compare_models(["VGG16", "InceptionV3"], data_dir, epochs=10,
               title="Exercise 5.10 - brain MRI tumor detection (no / yes)",
               save_as="../Output/exercise 5.10.png")
