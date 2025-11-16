# JPEG Compression from scratch

JPEG compression implemented in C++, using opencv4.

## Install
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```
At this point, `build/` should have a `myjpeg` binary. To install globally (i.e. in /usr/local/bin), run:
```bash
cd build/
sudo make install 
```

## Usage
```bash
myjpeg {image_file_path} [--qmi=N]
```
Here, `--qmi=N` gives the quantisation level. Valid values are {0,1,2,3} where 0 is no quanisation, and 1-3 are decreasing levels of quantisation (i.e. 3 should be clearer than 1)

## Example
`images/` includes test images. Note these are themselves JPEGs, and are thus already compressed. Here, we apply a more aggressive quantisation, so the compression will still be obvious.
```bash
myjpeg images/test_4.jpg --qmi=3
```