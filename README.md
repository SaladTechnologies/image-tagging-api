# image-tagging-api
An inference API for image tagging

## Download model weights

```bash
wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth -P ./src/data/
```


## Example Use

```bash
curl  -X POST \
  'http://localhost:8000/tag' \
  --header 'Accept: */*' \
  --form 'file=@/path/to/image.jpg'
```