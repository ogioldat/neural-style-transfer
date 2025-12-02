import keras


def VGGFeatureExtractor():
    vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    content_layer = "block4_conv2"
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]

    outputs = [vgg.get_layer(name).output for name in (style_layers + [content_layer])]

    return keras.Model(inputs=vgg.input, outputs=outputs, name="vgg_extractor")