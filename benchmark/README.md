# Benchmark Results

The current benchmark results are computed with Augraphy 8.20 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.
In order to get a same benchmark results with a different set of images, please run run_benchmarks.py using the following command:

```
python run_benchmarks.py --folder_path folder_path_with_images
```

|    Augmentation    |Images per second|Memory usage (MB)|
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |             0.37|           178.20|
|BindingsAndFasteners|            31.40|            21.43|
|BleedThrough        |             0.43|           684.69|
|BookBinding         |             0.11|           611.74|
|Brightness          |             5.75|           147.99|
|BrightnessTexturize |             2.16|           181.74|
|ColorPaper          |             5.07|           105.66|
|ColorShift          |             0.88|           114.26|
|DelaunayTessellation|             0.14|            60.00|
|DirtyDrum           |             0.96|           482.63|
|DirtyRollers        |             1.63|           249.52|
|Dithering           |             3.82|           126.82|
|DotMatrix           |             0.64|            80.77|
|Faxify              |             2.12|           126.74|
|Folding             |             2.62|            63.28|
|Gamma               |            34.39|            25.36|
|Geometric           |           183.89|            12.68|
|GlitchEffect        |             1.49|           126.94|
|Hollow              |             0.21|           343.17|
|InkBleed            |             4.00|           177.51|
|InkColorSwap        |             5.10|            51.99|
|InkMottling         |             7.60|            55.99|
|InkShifter          |             0.20|           426.89|
|Jpeg                |             6.38|            25.85|
|Letterpress         |             0.46|           158.10|
|LightingGradient    |             0.47|           638.31|
|LinesDegradation    |             1.51|           175.38|
|LowInkPeriodicLines |             6.19|            12.74|
|LowInkRandomLines   |           144.08|            12.74|
|LowLightNoise       |             0.32|           481.95|
|Markup              |             2.54|           154.12|
|NoiseTexturize      |             0.96|           249.36|
|NoisyLines          |             1.04|           446.89|
|PageBorder          |             0.56|           191.84|
|PatternGenerator    |             1.00|            51.53|
|ReflectedLight      |             0.08|           109.92|
|Scribbles           |             0.93|            99.10|
|SectionShift        |           154.45|            12.95|
|ShadowCast          |             0.86|            50.80|
|Squish              |             0.99|           443.70|
|SubtleNoise         |             1.82|           202.87|
|VoronoiTessellation |             0.09|            58.18|
|WaterMark           |             2.55|           390.55|
