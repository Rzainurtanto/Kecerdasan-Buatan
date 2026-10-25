# Exercise 5.7 Skin cancer image classification with VGG16 as the base model
# The book's HAM10000 dataset is 2.7 GB; this uses the smaller public Kaggle dataset
# "Skin Cancer: Malignant vs. Benign" (ISIC images, 2 classes, already split into train/test).
# The Kaggle notebooks for skin cancer use ResNet50 / DenseNet201; here VGG16 is compared with them.
# kagglehub downloads the dataset (no Kaggle login needed) into D:/AI/.kagglehub.
import os
from transfer_utils import kaggle_download, compare_models

path = kaggle_download("fanconic/skin-cancer-malignant-vs-benign")
train_dir = os.path.join(path, "train")
test_dir = os.path.join(path, "test")

# 300 training and 100 test images per class keep the run time reasonable on a CPU
compare_models(["VGG16", "ResNet50", "DenseNet201"], train_dir, test_dir, epochs=10,
               max_per_class=300, max_test_per_class=100,
               title="Exercise 5.7 - skin cancer (benign vs malignant)",
               save_as="../Output/exercise 5.7.png")
