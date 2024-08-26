### Notes

This implementation of Unet based on [fastMRI official code](https://github.com/facebookresearch/fastMRI/tree/master/models/unet).

The `F.interpolate` function was changed to `F.upsample` of older Pytorch version.
The `Dataset` class is modified so that down-sampling happens after cropping.
Other setting is almost identical to the original implementation.

For training, run
```
sh train.sh
```
For testing, run
```
sh test.sh
```
During testing, the output is scaled into the range [0, 255] to make a fair comparison with our proposed model.

There could be strange `segmentation fault`, especially when I tested the model during training. I haven't figured out why.
