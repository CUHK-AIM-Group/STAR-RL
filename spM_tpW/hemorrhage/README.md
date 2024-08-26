### Notes
To test the model on a custom dataset, a `.txt` file listing all the images is needed.

Put all images inside `test/`.

To test on this dataset of Axial T1 images exhibiting subacute intraparenchymal hemorrhage, please unzip `test.zip`, and run
```
python test --dataset hemorrhage --model path_to_the_model
```


The images are downloaded from 
[Radiopaedia.org](https://radiopaedia.org/cases/early-and-late-subacute-intracerebral-haemorrhage-on-mri-and-ct?lang=us) (rID: 55641).

We use the model trained on MICCAI using 30% mask.
Our model is able to reconstruct (improving PSNR and SSIM) all the images 
and the regions exhibiting pathology can be successfully restored, 
although the aliasing cannot be fully removed 
(partly due to the domain gap between the training data and test images).

This is an example of testing on out-of-distribution dataset. The output action distribution is very different from MICCAI.
