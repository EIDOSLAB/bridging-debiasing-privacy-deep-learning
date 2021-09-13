## Bridging the gap between debiasing and privacy for deep learning

![scatter](resources/scatter.png)

### Downloading checkpoints

Download pretrained checkpoints from: https://datacloud.di.unito.it/index.php/s/EixML38N7jBaZKa

### Training


- For training models regularized with REBIAS, Learned Mixin, RUBi and Vanilla models on MNIST use [train.sh](#train.sh), and refer to the [official REBIAS repository](https://github.com/clovaai/rebias)

- For training EnD regularized models on MNIST refer to the [EnD official repository](https://github.com/EIDOSlab/entangling-disentangling-bias)

- For training MNIST bias classifiers (attack) use [train_classifier.sh](#train_classifier.sh)

- For training bias classifiers (attack) on CelebA and IMDB refer to [train_celeba.py](#train_celeba.py) and [train_imdb.py](#train_imdb.py) (you must download the pretrained Vanilla and EnD checkpoints). If you want to train these models by yourself, refer to the hyperparm search described in the [EnD paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tartaglione_EnD_Entangling_and_Disentangling_Deep_Representations_for_Bias_Correction_CVPR_2021_paper.html)