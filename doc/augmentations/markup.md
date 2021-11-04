# Markup

The Markup augmentation mimics text strikethrough, underline, and highlight effect as if someone has drawn over text using a markup tool

**Example Usage:**
```python
augmentation=Markup(markup_type="underline",
                    num_lines_range=(2,4),
                    markup_thickness_range=(5,9),
                    markup_length_range=(1,1),
                    markup_color=(0,255,0),
                    p=1)
```

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `markup_type`    | Choice of markup "strikethrough", "highlight" or "underline".            |
| `num_lines_range` | Range to determining the number of lines to add strikethrough         |
| `markup_thickness_range` |  Range to determine the thickness of markup line      |
| `markup_length_range`| Pair of floats between 0 to 1 , to determine the length of strikethrough effect.                             |
| `markup_color`   |  BGR color tuple.                      |
| `p`             | The probability this augmentation will be applied.               |



Click here for an [example Colab Notebook](https://colab.research.google.com/drive/1ZT684tzqUMGsUTLrloPm2YansEUTs5nb?usp=sharing)
