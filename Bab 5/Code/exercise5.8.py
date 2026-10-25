# Exercise 5.8 Diabetic retinopathy classification with VGG16 as the base model
# Dataset from the book: Kaggle "diabetic-retinopathy-224x224-gaussian-filtered" (3,662 images,
# 5 classes: Mild, Moderate, No_DR, Proliferate_DR, Severe). Downloaded with kagglehub.
# VGG16 is compared with MobileNetV2 and DenseNet121.
from transfer_utils import kaggle_download, find_class_root, compare_models

path = kaggle_download("sovitrath/diabetic-retinopathy-224x224-gaussian-filtered")
data_dir = find_class_root(path)

# 120 images per class (the smallest class, Severe, has 193) -> balanced and quick on a CPU
compare_models(["VGG16", "MobileNetV2", "DenseNet121"], data_dir, epochs=10, max_per_class=120,
               title="Exercise 5.8 - diabetic retinopathy (5 classes)",
               save_as="../Output/exercise 5.8.png")
