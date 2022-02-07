# <a name="diploma">Diploma</a>

## Pose Guided Person Image Generation with Detailed Segmentation

## Table of contents
- [Description](#description)
- [Helpful links](#helpful)

## <a name="description"> Description </a> [↑](#diploma)

<p align="center">
  <img src="./docs/imgs/structure.jpeg" alt="structure" width="600"/>
</p>

Полученный алгоритм принимает на вход изображение и текст, по тексту определяет категорию, по категории генерирует скелет человека, далее находит фотографию из датасета, у которой скелет наиболее похож на сгенерируемый, и транслирует позу из полученного изображения на входное.

### Краткое описание реализации:
- сначала производится детализированная сегментация изображения человека, которая включает в себя сегментацию изображения человека и выделение одежды и лица
- по тексту определяется категория и с помощью условной GAN генерируется поза в рамках данной категории
- далее выбирается наиболее похожая по позе фотография человека
- два полученных изображения подаются на вход модели для трасляции изображений, которая возвращает исходное изображение с измененной позой

### Подробнее
- [презентация](./docs/presentation.pptx)
- [подробное описание работы](./docs/thesis.pdf)

## <a name="helpful"> Helpful links </a> [↑](#diploma)
- [DCGAN for segmentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Face GAN](https://github.com/IIGROUP/TediGAN)
- [Face GitHubs](https://github.com/ChanChiChoi/awesome-Face_Recognition)
- [Face Datasets](https://github.com/switchablenorms/CelebAMask-HQ)

- [Text-Guided Image Manipulation GAN](https://github.com/mrlibw/ManiGAN)

| Link  |  Comments |
|---|---|
| [SieveNet](https://github.com/levindabhi/SieveNet)  |  Try-On |
| [Virtual Try-on with Detail Carving](https://github.com/JDAI-CV/Down-to-the-Last-Detail-Virtual-Try-on-with-Detail-Carving)  |  Try-On |
| [Init Photo + Pose + Clothes](https://fashiontryon.wixsite.com/fashiontryon) |Try-On |

- [All about fashion](https://github.com/lzhbrian/Cool-Fashion-Papers)
- [Text to image](https://github.com/weihaox/awesome-image-translation/blob/master/content/multi-modal-representation.md#text-to-image)
