## SpMT 

PyTorch Code for "Semi-parametric Makeup Transfer via Semantic-aware Correspondence"

![cover](/imgs/cover.png)

### Requirements

+ Ubuntu 18.04
+ Anaconda (Python, Numpy, PIL, etc.)
+ PyTorch 1.7.1
+ torchvision 0.8.2

### Prepare data
#### For training phase:

```
opt.dataroot=MT-Dataset
├── images
│   ├── makeup
│   └── non-makeup
├── parsing
│   ├── makeup
│   └── non-makeup
├── makeup.txt
├── non-makeup.txt
```

#### For testing phase:
  
+ Use images of MT dataset:

```
opt.dataroot
├── images
│   ├── makeup
│   └── non-makeup
├── parsing
│   ├── makeup
│   └── non-makeup
├── makeup_test.txt
├── non-makeup_test.txt
```

+ Use arbitrary images:

```
opt.dataroot
├── images
│   ├── makeup
│   └── non-makeup
├── makeup_test.txt
├── non-makeup_test.txt
```

Facial masks of an arbitrary image will be obtained from the face parsing model (we borrow the model from https://github.com/zllrunning/face-parsing.PyTorch)

### Train:

```
python train.py --phase train
```

### Test:

1. Check the file 'options/demo_options.py', change the corresponding cofigs if needed

2. Create folder '/checkpoints/makeup_transfer/'

3. Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1aHwv1q5MfMfrCcweFObXITyF7dv4L-w8/view?usp=sharing) and put it into '/checkpoints/makeup_transfer/'

#### Use images of MT dataset:

```
python demo.py --demo_mode normal 
```
Notice:
+ Available demo mode: 'normal', 'interpolate', 'removal', 'multiple_refs', 'partly'

+ For part-specific makeup transfer(opt.demo_mode='partial'), make sure there are at least 3 reference images.

+ For interpolation between multiple references, make sure there are at least 4 reference images.

#### Use arbitrary images:

```
python demo_general.py  --beyond_mt
```

### Results

#### Shade-controllable

+ Interpolation from light to heavy

![shade1](/imgs/supplementary_controllable_shade1.png)

+ Interpolation between multiple references

![shade2](/imgs/supplementary_controllable_shade2.png)

#### Part-specific

Transfer different parts from different references

![part](/imgs/supplementary_controllable_part.png)

#### Makeup Removal

![removal](/imgs/supplementary_controllable_removal.png)

#### Comparison with Prior Arts

+ Normal Images

![normal](/imgs/supplementary_comparison_normal.png)

+ Wild Images

![wild1](/imgs/supplementary_comparison_wild1.png)
![wild2](/imgs/supplementary_comparison_wild2.png)


### Acknowledgments
This code borrows some function from [SPADE](https://github.com/NVlabs/SPADE) and [SCGAN](https://github.com/makeuptransfer/SCGAN)
