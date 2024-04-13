from waves import metric

""" An example for using `metric` module.

`text_image` folder structure is expected to be:

    test_image/
        |- clean/       ... Folder that stores clean images
        |- wm/          ... Folder that stores watermarked images
        |- attacked/    ... Folder that stores attacked watermarked images

There are some constrains to this function:
    - number of images in these folder are expected to be the same.
    - order of images in these folder are expected to be the same.
        e.g. if foo.png is the 5th image in clean/, then the 5th image in wm/ and attacked/ is
        expected to be the corresponding watermarked and attacked image of foo.png.
    - similarly, the prompt list should also be as the same order and in the image folder.

Ultimately, this function will be use internally and may or may not be accessible to user.
If the function is available to user when released, the constrains will be fixed and optimized.
"""

result = metric.generate_metrics(
    'test_image/',
    metric.TEST_PROMPT, # Subject to change
    self_metric_data_source=metric.DataOption.NONE,
    cmp_metric_data_source=metric.DataOption.WATERMARKED_VS_ATTACKED,
    batch_size=2,
)

print(result)