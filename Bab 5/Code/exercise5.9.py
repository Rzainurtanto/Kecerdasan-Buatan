# Exercise 5.9 COVID-19 chest X-ray detection: MobileNet instead of VGG16 as the base model
# Dataset from the book: Kaggle "covid19-chest-xray-image-dataset" (covid / normal).
# The Kaggle project uses VGG16; both are trained here so the performance can be compared.
from transfer_utils import kaggle_download, find_class_root, compare_models

path = kaggle_download("alifrahman/covid19-chest-xray-image-dataset")
data_dir = find_class_root(path)

compare_models(["VGG16", "MobileNet"], data_dir, epochs=10,
               title="Exercise 5.9 - COVID-19 chest X-ray (covid vs normal)",
               save_as="../Output/exercise 5.9.png")
