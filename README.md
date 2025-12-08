# PyFire
Simple Fire in Python

![Number Fire](https://github.com/rykrr/PyFire/blob/9cae954d3deb13de7b199fcf8162764385f7b6c4/number_demo.gif)

```
usage: pyfire.py [-h] [-t TIMING] [-s STRENGTH] [-a ALPHA] [-d DECAY] [-b BIAS] [-f FLATTEN] [-c CLIP] [--margin MARGIN] [--mode MODE]
                 [--buffer BUFFER] [--cache CACHE]

options:
  -h, --help            show this help message and exit
  -t TIMING, --timing TIMING
                        Frame time in seconds
  -s STRENGTH, --strength STRENGTH
                        Maximum fire strength
  -a ALPHA, --alpha ALPHA
                        Constant applied to kernel for decay
  -d DECAY, --decay DECAY
                        Decay coefficient; Lower values make colder regions decay faster
  -b BIAS, --bias BIAS  Balance between random (0) and shaped (1) flames
  -f FLATTEN, --flatten FLATTEN
                        Controls how flat the shaped flame is; lower is flatter
  -c CLIP, --clip CLIP  Remove the bottom n rows (they're too chaotic)
  --margin MARGIN       Margins from side
  --mode MODE           Choose character set
  --buffer BUFFER       Render buffer size
  --cache CACHE         Use a fixed set of ignition seeds
```
