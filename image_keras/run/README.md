# Image Keras Run

## Slice Image

Slice Image for a image file and a folder contains images.
(Currently, it may be not work for options `not_discard_rest_vertical_tile` and `not_discard_rest_horizontal_tile`)

![Slice Image Size Guide](../../doc/slice_image_guide.png)

```shell
python image_keras/run/run_slice_image.py \
    --full_image_path tests/test_resources/lenna.png \
    --tile_size=256 \
    --inbox_size=64 \
    --stride_size=64 \
    --as_gray \
    --target_folder=temp
```

```shell
python image_keras/run/run_slice_image.py \
    --full_image_path tests/test_resources/lenna.png \
    --tile_size=256 \
    --inbox_size=60 \
    --stride_size=60 \
    --add_same_padding \
    --as_gray \
    --target_folder=temp
```

## Resize Image

Resize Image for a image file and a folder contains images.

```shell
python image_keras/run/run_resize_image.py \
    --full_image_path temp/lenna_00_00.png \
    --target_size=128 \
    --target_folder=temp2
```

```shell
python image_keras/run/run_resize_image.py \
    --full_image_path temp \
    --target_size=128 \
    --target_folder=temp3
```
