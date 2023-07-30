# Benchmark Results

The current benchmark results are computed with Augraphy 8.20 and Tobacco3482 dataset (resume subset with a total of 120 images). It is evaluated with a 2 cores machine - Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.
In order to get a same benchmark results with a different set of images, please run run_benchmarks.py using the following command:

```
python run_benchmarks.py --folder_path folder_path_with_images
```

|    Augmentation    |Images per second|Memory usage (MB)|
|--------------------|----------------:|----------------:|
|BadPhotoCopy        |             0.38|           178.20|
|BindingsAndFasteners|            29.34|            21.43|
|BleedThrough        |             0.39|           684.69|
|BookBinding         |             0.08|           611.74|
|Brightness          |             5.93|           147.99|
|BrightnessTexturize |             2.12|           181.74|
|ColorPaper          |             5.12|           105.66|
|ColorShift          |             0.87|           114.26|
|DelaunayTessellation|             0.14|            60.00|
|DirtyDrum           |             0.82|           482.63|
|DirtyRollers        |             1.58|           249.52|
|Dithering           |             3.96|           126.82|
|DotMatrix           |             0.22|            80.53|
|Faxify              |             2.02|           126.75|
|Folding             |             2.85|            63.27|
|Gamma               |            34.27|            25.36|
|Geometric           |           189.37|            12.68|
|GlitchEffect        |             1.48|           126.94|
|Hollow              |             0.21|           343.17|
|InkBleed            |             3.93|           177.51|
|InkColorSwap        |             5.22|            51.99|
|InkMottling         |             7.78|            55.99|
|InkShifter          |             0.17|           426.89|
|Jpeg                |             6.32|            25.85|
|Letterpress         |             0.45|           158.10|
|LightingGradient    |             0.44|           638.31|
|LinesDegradation    |             1.69|           175.38|
|LowInkPeriodicLines |             6.47|            12.74|
|LowInkRandomLines   |           114.02|            12.74|
|LowLightNoise       |             0.30|           481.95|
|Markup              |             2.97|           154.12|
|NoiseTexturize      |             0.73|           249.36|
|NoisyLines          |             0.86|           446.76|
|PageBorder          |             0.32|           201.22|
|PatternGenerator    |             1.68|            51.52|
|ReflectedLight      |             0.08|           109.97|
|Scribbles           |             1.00|            97.13|
|SectionShift        |           129.82|            12.96|
|ShadowCast          |             0.95|            50.79|
|Squish              |             0.26|           465.31|
|SubtleNoise         |             1.68|           202.87|
|VoronoiTessellation |             0.08|            57.48|
|WaterMark           |             2.70|           352.34|
