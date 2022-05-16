# vgg-style-transfer
Neural style transfer with a VGG19 network. 

## Neural Style Transfer

Neural style transfer (NST) is the process of combining the style of one image with the content of another image to produce a generated image.  NST is among the best examples of the power and creative capability of machine learning, having produced many popular and eye-catching examples of images in the style of famous paintings and artists. 

In a typical NST approach, a pre-trained convolutional neural network configured to detect low and high-level features from images is used to extract the style from a so-called “style” image, and the content from a so-called “content” image.  The extracted style and content are combined to produce a “generated” image, which is intended to exhibit the extracted content in the extracted style.

## VGG19 Network

In this implementation of NST, a VGG19 neural network pre-trained on ImageNet is used to extract the style and content of input images. The VGG19 network comprises 16 convolutional layers, and when pre-trained on ImageNet, provides a suitable mechanism for extracting the style and content of input images.  To extract style/content, an input image is forward propagated throughout the entire VGG19 network, and output activations from a select subset of layers are obtained.

### Content Extraction

The final convolutional layer of the VGG19 network – named “block5_conv4” – is used to extract an encoding of the content in an input image.  Being at the end of the network, this layer detects higher-level, more complex features in images and is thus suitable for obtaining content encodings of input images.

### Style Extraction

Five layers evenly distributed throughout the extent of the VGG19 network are used to extract an encoding of the style of an input image: 
- 'block1_conv1'
- 'block2_conv1' 
- 'block3_conv1'
- 'block4_conv1'
- 'block5_conv1'

Being evenly distributed across the network, these “style” layers provide lower-level and higher-level encodings of image style. 

## Optimization and Image Generation

To produce a generated image combining the content of a content image and style of a style image, a cost function is reduced that is a weighted sum of individual content and style cost functions.

The content cost function measures the difference between (1) output activations from the content layer, produced by forward propagating the content image through the VGG19 network, and (2) output activations from the content layer, produced by forward propagating the generated image through the VGG19 network.  

The style cost function measures the difference between (1) a Gram matrix computed based on output activations from a single style layer produced by forward propagating the style image through the VGG19 network, and (2) a Gram matrix computed based on output activations from the single style layer produced by forward propagating the generated image through the VGG19 network.  This Gram matrix difference is summed over the five style layers to compute an overall style cost between the style and generated images.

To assist the style transfer process, the generated image is initialized by taking the content image and adding random uniform noise.

## Script Use

`script.py` expects a content image at `images/content.jpg` and a style image at `images/style.jpg`. Input images are resized to 400x400.  Generated output images are saved at `output_images`.

By default, `script.py` loads saved weights of a VGG19 network pre-trained on ImageNet from `model/`.  If weights are not available from disk, a pre-trained VGG19 instance can be loaded via `get_vgg_model()`, either from the internet (by setting `mode='download'`) or from a filepath (by setting `mode='local'`, and specifying `path`).

`load_images()` loads and resizes content and style images.

`initialize_generated_image()` initializes a generated image by adding random uniform noise to the content image.

`get_vgg_encodings()` instantiates a custom Keras model that outputs encodings from select layers of the VGG19 network defined by `ENCODING_LAYERS`. This custom model is used to obtain style and content encodings of input images.

`transfer_style_loop()` performs style transfer for a number of epochs set by `epochs`.  At regular intervals, a generated image produced at the given epoch is saved in `output_images/`.

By default, style transfer is performed for 25,000 epochs with a learning rate of 0.001. Higher epoch numbers and lower learning rates are recommended for best results. 

`transfer_style_loop()` calls `transfer_style_step()`, which performs a single step of style transfer. This function utilizes the `apply_gradients()` method, which is from older versions of Tensorflow (see v1.15.0 documentation for `tf.keras.optimizers.Adam`).

## Limitations and Future Development

While desirable results can be obtained with default settings, adjustment of various parameters should be explored, such as which VGG19 layers are specified in `ENCODING_LAYERS` as the layers used to extract style/content, and their weights. Weight adjustment could also be performed for the weightings applied to the style and content costs in computing the total cost.

While the addition of noise in initializing the generated image appears to assist style transfer, such noise can result in degraded image quality in generated images produced in later epochs. Alternative approaches are possible that could mitigate this issue. 
