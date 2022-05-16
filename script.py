from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Settings 
IMG_SIZE = 400                                 # resize content/style images to (IMG_SIZE, IMG_SIZE)
VGG_NETWORK_PATH = 'model/'                    # path to saved pre-trained VGG19 network
CONTENT_IMG_PATH = 'images/content.jpg'        # path to content image to transfer style to
STYLE_IMG_PATH = 'images/style.jpg'            # path to style image to transfer style from
OUTPUT_PATH = 'output_images/'                 # path to folder in which to save generated images
CONTENT_LAYER = 'block5_conv4'                 # VGG19 network layer used to obtain image content

ENCODING_LAYERS = [('block1_conv1', 0.2),      # VGG19 network layer used to obtain image style; weight for layer
                ('block2_conv1', 0.2),         # VGG19 network layer used to obtain image style; weight for layer
                ('block3_conv1', 0.2),         # VGG19 network layer used to obtain image style; weight for layer
                ('block4_conv1', 0.2),         # VGG19 network layer used to obtain image style; weight for layer
                ('block5_conv1', 0.2),         # VGG19 network layer used to obtain image style; weight for layer
                (CONTENT_LAYER, 1)]            # VGG19 network layer used to obtain image content; weight for layer

LEARN_RATE = 0.001                             # Default style transfer learning rate
EPOCHS = 25_000                                # Default number of style transfer epochs


# Functions
def get_vgg_model(mode='download', path=None):
    '''
    Build pre-trained VGG19 network with weights obtained via download or from filepath.

    Arguments
    ---------
    mode: string
        'download' downloads VGG19 weights (trained on imagenet) from internet
        'local' retrieves VGG19 weights from filepath
    path: string/filepath
        filepath to saved VGG19 weights

    Returns
    -------
    vgg_model: Keras model
        pre-trained VGG19 network
    '''

    if mode == 'download':
        vgg_model = tf.keras.applications.VGG19(include_top=False,
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   weights='imagenet')

    elif mode == 'local':
        vgg_model = tf.keras.applications.VGG19(include_top=False,
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   weights=path)

    return vgg_model


def load_images():
    '''
    Load and resize content image from CONTENT_IMG_PATH and style image from STYLE_IMG_PATH.

    Returns
    -------
    content_img: tensor of shape (m, IMG_SIZE, IMG_SIZE, channels)
        resized content image to transfer style to
    style_img: tensor of shape (m, IMG_SIZE, IMG_SIZE, channels)
        resized style image to obtain style from
    '''

    content_img = Image.open(CONTENT_IMG_PATH).resize((IMG_SIZE, IMG_SIZE))
    content_img = tf.expand_dims(content_img, axis=0)   # add batch dimension and convert to EagerTensor

    print(f'Content image obtained at: {CONTENT_IMG_PATH}\
            \nContent image resized to {IMG_SIZE, IMG_SIZE}')

    style_img = Image.open(STYLE_IMG_PATH).resize((IMG_SIZE, IMG_SIZE))
    style_img = tf.expand_dims(style_img, axis=0)   # add batch dimension and convert to EagerTensor

    print(f'Style image obtained at: {STYLE_IMG_PATH}\
            \nStyle image resized to {IMG_SIZE, IMG_SIZE}')

    print(f'Content image dtype: {content_img.dtype}\
            \nStyle image dtype: {style_img.dtype}')

    return content_img, style_img


def initialize_generated_image(content_img, noise_magnitude=0.20):
    '''
    Initialize generated image by adding noise to, and normalizing pixel values of, content image.

    Arguments
    ---------
    content_img: tensor of shape (m, IMG_SIZE, IMG_SIZE, channels)
        content image used to initialize generated image
    noise_magnitude: float
        magnitude of random uniform noise to add to content image

    Returns
    -------
    generated_img: ResourceVariable of shape (m, IMG_SIZE, IMG_SIZE, channels)
        initialized generated image to transfer style to
    '''

    # Convert content_img from uint8 to float32, as tf.random.uniform() outputs noise of dtype float32
    # convert_image_dtype() normalizes generated_img pixel values
    generated_img = tf.image.convert_image_dtype(content_img, tf.float32)

    noise = tf.random.uniform(generated_img.shape, (noise_magnitude * -1), noise_magnitude)
    generated_img = tf.add(generated_img, noise)

    # Maintain normalization of generated_img pixel values
    generated_img = tf.clip_by_value(generated_img, 0.0, 1.0)

    # This step is necessary to successfully call apply_gradients() in transfer_style_step()
    return tf.Variable(generated_img)


def get_vgg_encodings(vgg_model, encoding_layers):
    '''
    Instantiate custom Keras model that forward propagates images through VGG19 network and outputs activations
    from selected VGG19 network layers, defined by encoding_layers, to obtain style/content encodings of images.
    
    Arguments
    ---------
    vgg_model: Keras model
        pre-trained VGG19 network
    encoding_layers: list
        VGG19 network layers used to obtain style/content encodings of input images

    Returns
    -------
    model: Keras model
        custom Keras model that outputs activations from layes defined by encoding_layers
    '''

    outputs = []

    for layer in encoding_layers:
        outputs.append(vgg_model.get_layer(name=layer[0]).output)

    model = tf.keras.Model(vgg_model.input, outputs)

    return model


def get_style_cost(encoded_style_img, encoded_generated_img, encoding_layers):
    '''
    Compute style cost as a function of the difference between the (1) output activations produced by 
    forward propagating the style image through custom VGG19 network, and (2) output activations produced
    by forward propagating the generated image through custom VGG19 network. Output activations are those
    from the style layers in encoding_layers.

    Arguments
    ---------
    encoded_style_img: list
        output activations, produced by forward propagating style image through VGG19 network, from the layers
        in encoding_layers
    encoded_generated_img: list
        output activations, produced by forward propagating generated image through VGG19 network, from the layers
        in encoding_layers
    encoding_layers: list
        layers of VGG19 network to obtain activations from for encoding style and/or content of an input image

    Returns
    -------
    J_style: tensor
        style cost computed as function of encoded style image and encoded generated image

    '''

    J_style = 0

    '''
    encoded_style_img, encoded_generated_img are lists of 6 layer encodings - i.e., the 
    "full" encoding consisting of five style layer encodings and a content layer encoding.
    To compute style cost for a single layer, select the first five layers ([:-1]), and the 
    [i]th element.
    '''

    style_img_style = encoded_style_img[:-1]
    generated_img_style = encoded_generated_img[:-1]

    # Compute style cost for each individual style encoding layer, looping over all style encoding layers
    for layer, weight in zip(range(len(style_img_style)), encoding_layers):

        style_layer = style_img_style[layer]
        generated_layer = generated_img_style[layer]

        height = style_layer.shape[1]
        width = style_layer.shape[2]
        channels = style_layer.shape[-1]

        # Reshape from 4D to 2D tensor, and put channels first
        style_layer = tf.reshape(style_layer, [-1, channels])
        style_layer = tf.transpose(style_layer, perm=[1,0])
        generated_layer = tf.reshape(generated_layer, [-1, channels])
        generated_layer = tf.transpose(generated_layer, perm=[1,0])

        # Compute Gram matrices 
        style_gram_matrix = tf.matmul(style_layer, tf.transpose(style_layer))
        generated_gram_matrix = tf.matmul(generated_layer, tf.transpose(generated_layer)) 

        # Compute loss
        const = 1.0 / (4.0 * ((channels)**2) * ((height * width)**2))
        diff = tf.subtract(style_gram_matrix, generated_gram_matrix)

        J_layer = tf.multiply(const, tf.reduce_sum(tf.square(diff)))

        J_style += weight[1] * J_layer

    return J_style


def get_content_cost(encoded_content_img, encoded_generated_img):
    '''
    Compute content cost as a function of the difference between the (1) output activation produced by 
    forward propagating the content image through the VGG19 network, and (2) output activation produced
    by forward propagating the generated image through the VGG19 network. Each output activation is from 
    the content layer defined in encoding_layers.

    Arguments
    ---------
    encoded_content_img: list
        output activation, produced by forward propagating content image through VGG19 network, from content layer
        in encoding_layers
    encoded_generated_img: list
        output activation, produced by forward propagating generated image through VGG19 network, from content layer
        in encoding_layers

    Returns
    -------
    J_content: tensor
        content cost computed as function of encoded content image and encoded generated image
    '''

    # Get content encoding of content and generated images
    content_img_content = encoded_content_img[-1]
    generated_img_content = encoded_generated_img[-1]

    height = content_img_content.shape[1]
    width = content_img_content.shape[2]
    channels = content_img_content.shape[-1]

    # Reshape from 4D to 3D tensor, and put channels first
    content_img_content = tf.reshape(content_img_content, shape=[1, -1, channels])
    content_img_content = tf.transpose(content_img_content, perm=[0, 2, 1])
    generated_img_content = tf.reshape(generated_img_content, shape=[1, -1, channels])
    generated_img_content = tf.transpose(generated_img_content, perm=[0, 2, 1])

    # Compute content cost
    const = (1.0 / (4.0 * height * width * channels))
    J_content = tf.multiply(const, tf.reduce_sum(tf.square(tf.subtract(content_img_content, generated_img_content))))
    
    return J_content


def get_total_cost(J_style, J_content, content_weight=10, style_weight=40):
    '''
    Compute total cost as weighted sum of style and content cost.

    Arguments
    ---------
    J_style: tensor
        style cost
    J_content: tensor
        content cost
    content_weight: int
        weighting applied to content cost
    style_weight: int
        weighting applied to style cost

    Returns
    -------
    J: tensor
        total cost
    '''

    content_weight = tf.cast(content_weight, tf.float32)
    style_weight = tf.cast(style_weight, tf.float32)
    
    J = tf.multiply(content_weight, J_content) + tf.multiply(style_weight, J_style)

    return J


@tf.function()
def transfer_style_step(encoded_content_img, encoded_style_img, generated_img, encoding_layers):
    '''
    Perform single step of style transfer. In one step, produce generated image based on content
    from content image and style from style image. Beginning with initialized generated image, apply gradients 
    computed with respect to total cost to produce generated image, and maintain pixel value normalization.

    Arguments
    ---------
    encoded_content_img: list
        output activations, produced by forward propagating content image through VGG19 network, from layers
        in encoding_layers
    encoded_style_img: list
        output activations, produced by forward propagating style image through VGG19 network, from layers
        in encoding_layers
    generated_img: ResourceVariable/tensor
        initialized generated image
    encoding_layers: list
        layers of VGG19 network to obtain activations from for encoding style and/or content of an input image
    '''

    with tf.GradientTape() as tape:

        encoded_generated_img = encoding_model(generated_img)

        J_style = get_style_cost(encoded_style_img, encoded_generated_img, encoding_layers)
        J_content = get_content_cost(encoded_content_img, encoded_generated_img)

        J = get_total_cost(J_style, J_content)

    gradient = tape.gradient(J, generated_img)

    # apply_gradients is from a prior version of Tensorflow (see v1.15.0 documentation for tf.keras.optimizers.Adam)
    optimizer.apply_gradients([(gradient, generated_img)])
    generated_img.assign(tf.clip_by_value(generated_img, 0.0, 1.0))


def transfer_style_loop(epochs=EPOCHS):
    '''
    Perform a number of steps of style transfer defined by epochs. Save and display generated images
    at every 1/100 of epochs.

    Arguments
    ---------
    epochs: int
        number of epochs to perform style transfer
    '''

    for i in range(epochs):
        transfer_style_step(encoded_content_img, encoded_style_img, generated_img, ENCODING_LAYERS)
        
        # Print epoch number at each epoch
        print(f'Epoch number: {i}')

        # At every 1/100th of epochs, save generated image at that epoch
        # For 25,000 epochs, this will happen every 250 epochs, resulting in 100 saved images
        if i % (epochs/100) == 0:
            img = generated_img[0] * 255
            img = np.array(img, dtype=np.uint8)
            img = Image.fromarray(img)
            img.save(OUTPUT_PATH + f'image_{i}.jpg')
        


############################
########---Script---########
############################

# Load and resize content and style images
content_img, style_img = load_images()

# Normalize content and style images, and convert from uint8 to float32
content_img = tf.image.convert_image_dtype(content_img, tf.float32)
style_img = tf.image.convert_image_dtype(style_img, tf.float32)

# Initialize generated image by adding random noise to content image
generated_img = initialize_generated_image(content_img)

# Get pre-trained VGG19 network; use get_vgg_model() if pre-trained VGG19 unavailable from disk
vgg_model = load_model(VGG_NETWORK_PATH)
vgg_model.trainable = False

# Get custom encoding model that outputs style and/or content encodings of input images 
# using the layers from vgg_model defined by ENCODING_LAYERS
encoding_model = get_vgg_encodings(vgg_model, ENCODING_LAYERS)

# Get style/content encodings of content and style images
encoded_content_img = encoding_model(content_img)
encoded_style_img = encoding_model(style_img)

# Define optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

# Perform style transfer for number of epochs defined by epochs
transfer_style_loop(epochs=EPOCHS)





