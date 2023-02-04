# Visualize ResNet embedings with UMAP

I took my personal backup (mostly family photos, some random scans, etc), extracted the embeddings from [ResNet][ref_resnet] (pen-ultimate layer activations), and then plotted them with the [UMAP algorithm][ref_umap] and [bokeh library][ref_bokeh]

This was a one-afternoon learning exercise while doing the awesome [Fast AI course][ref_fastai]. The results are quite fun. ResNet returns 512 feature and UMAP maps those features into 2D plane, preserving distances as much as possible. The outliers show that the embedings "make sense", clustering the images that are similar to other ones in the cluster, and different from those outside the cluster.

[ref_resnet]:https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
[ref_umap]:https://umap-learn.readthedocs.io/en/latest/basic_usage.html
[ref_bokeh]:https://bokeh.org/
[ref_fastai]:https://course.fast.ai/

Here is a cluster of too dark pics:

![img](gifs/emb-1.gif)

Kids on bike and red background:

![img](gifs/emb-2.gif)

Some seaside:

![img](gifs/emb-3.gif)

Random landscapes:

![img](gifs/emb-4.gif)

Bathroom hardware store:

![img](gifs/emb-5.gif)

# To recreate

1. Create thumbnails of the pictures with [`process_photos.py`](process_photos.py):

```
./process_photos.py -d /Volumes/MyStuff -o output/torch_thumbs -s 224 >> output/torch_thumbs.meta

# To resume interrupted session use:
#
# ./process_photos.py -d  /Volumes/MyStuff -o output/torch_thumbs -s 224 -l output/torch_thumbs.meta  | tee -a output/torch_thumbs.meta
```

2. Extract features with resnet18 model with [`embeddings.py`](embeddings.py). I tried also resnet34, but the results were actually worse.

```
find output/torch_thumbs -name \*.jpg | ./embeddings.py -o  output/features_resnet18 -m resnet18 -
```

3. Cast the embeddings from 512 feature space to good old 2d space with umap-learn.  Visualize the results with `bokeh`.  Running UMAP and the visualization is done from a Jupyter notebook [`visualize.ipynb`](visualize.ipynb).

4. To show thumbnails when you hover over a point you need to run python3 -m http.server to serve the images.

Voila!

[HN]()

[T]()
