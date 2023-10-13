# Benchmark Results

The current benchmark results are computed with Augraphy 8.2 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz. The mask is using a binarized version of image. For keypoints, it is using 1000 random points in the image and for bounding boxes, 100 random bounding boxes with random size are used. In order to get a same benchmark results with a different set of images, please run run_benchmarks.py using the following command:

```
python run_benchmarks.py --folder_path folder_path_with_images
```

|    Augmentation    |Img/sec|Memory(MB)|Img/sec(mask)|Memory(MB)(mask)|Img/sec(keypoints)|Memory(MB)(keypoints)|Img/sec(bounding boxes)|Memory(MB)(bounding boxes)|
|--------------------|------:|---------:|------------:|---------------:|-----------------:|--------------------:|----------------------:|-------------------------:|
|BadPhotoCopy        |   0.17|    202.81|         0.12|          215.60|              0.13|               204.55|                   0.12|                    216.29|
|BindingsAndFasteners|  28.21|     21.08|        29.75|           21.02|             36.39|                21.02|                  29.09|                     21.23|
|BleedThrough        |   0.39|    684.69|         0.39|          684.69|              0.39|               684.69|                   0.39|                    684.69|
|BookBinding         |   0.09|    611.71|         0.08|          616.87|              0.09|               611.74|                   0.09|                    611.67|
|Brightness          |   4.92|    147.99|         4.90|          147.99|              4.95|               147.99|                   5.07|                    147.99|
|BrightnessTexturize |   1.83|    181.74|         1.83|          181.74|              1.84|               181.74|                   1.86|                    181.74|
|ColorPaper          |   4.83|    105.66|         4.99|          105.66|              4.96|               105.66|                   4.96|                    105.66|
|ColorShift          |   0.79|    126.94|         0.82|          126.94|              0.77|               126.94|                   0.76|                    126.94|
|DelaunayTessellation|   0.11|     60.41|         0.12|           60.29|              0.11|                60.36|                   0.10|                     60.37|
|DirtyDrum           |   0.83|    482.51|         0.94|          481.56|              0.90|               481.68|                   0.92|                    481.52|
|DirtyRollers        |   1.47|    249.55|         1.77|          249.43|              1.80|               249.43|                   1.78|                    249.43|
|Dithering           |   3.39|    126.82|         3.66|          126.80|              3.66|               126.81|                   3.79|                    126.80|
|DotMatrix           |   0.53|     80.75|         0.57|           80.52|              0.57|                80.52|                   0.57|                     80.52|
|Faxify              |   1.37|    138.34|         1.43|          141.28|              1.41|               136.95|                   1.27|                    149.24|
|Folding             |   3.18|     57.50|         1.24|           60.40|              3.44|                57.60|                   3.55|                     57.20|
|Gamma               |  29.26|     25.36|        26.90|           25.36|             28.23|                25.36|                  32.03|                     25.36|
|Geometric           | 135.75|     12.68|       137.64|           12.68|            145.80|                12.68|                 127.10|                     12.68|
|GlitchEffect        |   1.14|    132.35|         1.10|          132.65|              1.03|               134.14|                   1.05|                    134.11|
|Hollow              |   0.17|    343.17|         0.17|          343.17|              0.17|               343.17|                   0.17|                    343.17|
|InkBleed            |   3.23|    177.51|         3.19|          177.51|              3.17|               177.51|                   3.16|                    177.51|
|InkColorSwap        |   3.47|     51.99|         3.58|           51.99|              3.51|                51.99|                   3.61|                     51.99|
|InkMottling         |   5.41|     55.99|         5.47|           55.99|              5.49|                55.99|                   5.39|                     55.99|
|InkShifter          |   0.17|    426.86|         0.15|          426.43|              0.17|               426.78|                   0.17|                    426.58|
|Jpeg                |   5.55|     25.87|         5.60|           25.86|              5.52|                25.87|                   5.66|                     25.87|
|Letterpress         |   0.35|    135.71|         0.33|          140.72|              0.34|               137.25|                   0.34|                    136.31|
|LightingGradient    |   0.37|    638.31|         0.38|          638.30|              0.39|               638.30|                   0.40|                    638.30|
|LinesDegradation    |   1.28|    174.76|         1.27|          174.93|              1.28|               174.92|                   1.31|                    174.59|
|LowInkPeriodicLines |   5.17|     12.75|         5.26|           12.75|              5.56|                12.75|                   5.10|                     12.75|
|LowInkRandomLines   |  91.52|     12.75|        86.12|           12.75|             87.58|                12.75|                  98.28|                     12.75|
|LowLightNoise       |   0.27|    481.95|         0.28|          481.95|              0.27|               481.95|                   0.27|                    481.95|
|Markup              |   2.33|    161.88|         2.41|          158.13|              2.58|               146.27|                   2.60|                    147.53|
|NoiseTexturize      |   0.83|    249.36|         0.85|          249.36|              0.80|               249.36|                   0.82|                    249.36|
|NoisyLines          |   0.89|    446.65|         0.83|          448.43|              0.86|               447.88|                   0.85|                    448.52|
|PageBorder          |   0.49|    193.95|         0.49|          188.46|              0.48|               188.30|                   0.49|                    192.04|
|PatternGenerator    |   0.76|     51.53|         0.76|           51.50|              0.74|                51.50|                   0.74|                     51.50|
|ReflectedLight      |   0.06|    109.90|         0.06|          109.82|              0.06|               109.88|                   0.06|                    110.02|
|Scribbles           |   1.11|     94.73|         0.86|           96.90|              0.87|                96.96|                   0.86|                    100.55|
|SectionShift        | 117.15|     12.96|       101.71|           13.02|            107.88|                12.95|                 115.22|                     12.96|
|ShadowCast          |   0.75|     50.79|         0.68|           50.80|              0.68|                50.80|                   0.74|                     50.80|
|Squish              |   0.72|    450.44|         0.72|          450.79|              0.73|               450.83|                   0.76|                    451.00|
|SubtleNoise         |   1.44|    215.55|         1.47|          215.55|              1.47|               215.55|                   1.48|                    215.55|
|VoronoiTessellation |   0.07|     58.07|         0.07|           57.74|              0.07|                57.89|                   0.07|                     58.13|
|WaterMark           |   2.09|    363.62|         1.86|          404.62|              1.78|               409.50|                   2.02|                    380.21|
