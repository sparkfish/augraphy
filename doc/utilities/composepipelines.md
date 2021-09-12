# ComposePipelines

The ComposePipelines class provides a means of composing two pipelines into one.

**Example Usage:**

```python
pipeline1 = AugraphyPipeline(ink_layer1, paper_layer1, post_layer1)

pipeline2 = AugraphyPipeline(ink_layer2, paper_layer2, post_layer2)

composedPipeline = ComposePipelines(pipeline1, pipeline2)
```

| Parameter | Description                           |
|-----------|---------------------------------------|
| `first`   | The first AugraphyPipeline to apply.  |
| `second`  | The second AugraphyPipeline to apply. |
