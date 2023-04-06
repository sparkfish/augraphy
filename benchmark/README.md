# Benchmark Results

The current benchmark results are computed with Augraphy 8.20 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.
In order to get a same benchmark results with a different set of images, please run run_benchmarks.py using the following command:

```
python run_benchmarks.py --folder_path folder_path_with_images
```

|    Augmentation    |Images per second|Memory usage (MB)|
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |             0.31|           138.25|
|BindingsAndFasteners|            26.11|            20.81|
|BleedThrough        |             0.27|           684.69|
|BookBinding         |             0.06|           683.50|
|Brightness          |             3.31|           147.99|
|BrightnessTexturize |             1.25|           181.74|
|ColorPaper          |             3.14|           105.66|
|DirtyDrum           |             0.55|           481.19|
|DirtyRollers        |             1.51|           173.45|
|Dithering           |             2.27|           126.82|
|Faxify              |             0.96|           142.97|
|Folding             |             5.73|            67.75|
|Gamma               |            25.39|            25.36|
|Geometric           |             5.24|            40.85|
|InkBleed            |             0.67|           294.73|
|Jpeg                |             3.59|            25.86|
|Letterpress         |             0.23|           158.60|
|LightingGradient    |             0.34|           638.31|
|LowInkPeriodicLines |             2.94|            12.74|
|LowInkRandomLines   |            71.05|            12.74|
|Markup              |             0.71|           533.16|
|NoiseTexturize      |             0.53|           249.36|
|PageBorder          |             0.52|           465.19|
|PencilScribbles     |             1.15|           138.13|
|SubtleNoise         |             1.03|           202.87|
|WaterMark           |             1.19|           373.41|
