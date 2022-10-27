# Examples of images and audio files

There are a few examples of images and audio files under this subdirectory.
One can generate more with expressions such as these:

```sh
gft_cat_data --data H:beans --split train --eqn 'classify_images:labels ~ image_file_path' | head
```

```sh
gft_cat_data --data H:timit_asr --split train --eqn 'ctc:text ~ file'
```

