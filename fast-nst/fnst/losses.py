import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Implementations for Neural Style Transfer content / style loss functions.

# Calculates the feature correlation matrix for a 3D tensor
# (no batch dimension)
def feat_gram_matx_3d(layer, name='feat_gram_matx_3d'):
    layer_shape = layer.shape.as_list()
    layer_area = tf.to_float(layer_shape[0] * layer_shape[1])
    layer_volume = tf.to_float(layer_area * layer_shape[2])

    #feat_matx = tf.reshape(layer, [layer_area, layer_shape[2]])
    #feat_trans = tf.transpose(feat_matx)

    #return tf.matmul(feat_matx, feat_trans) / layer_volume

    return tf.tensordot(layer, layer, axes=[[0,1], [0,1]], name=name) / layer_volume

# Computes layer style losses given two feature maps from two layers
# Note that both layers should be 3D tensors (i.e. no batch dimensions)
def layer_style_loss(generated_layer, style_gram_matrix, name='layer_style_loss'):
    generated_gram_matrix = feat_gram_matx_3d(generated_layer, name=name+'-generated-tdot')
    #e_unscaled = tf.reduce_sum(tf.square(generated_gram_matrix - style_gram_matrix))

    matx_diff = generated_gram_matrix - style_gram_matrix # 2D difference between matrices

    return tf.norm(matx_diff, 'fro', axis=(0, 1)) # Scalar frobenius norm




def batched_gram_matx(args, layers):
    batch, height, width, depth = layers.shape.as_list()

    if not args.enable_transpose_style_loss:
        unbatched_layers = tf.unstack(layers, axis=0, num=batch)
        unbatched_matrices = []
        for layer in unbatched_layers:
            unbatched_matrices.append(feat_gram_matx_3d(layer))

        return tf.stack(unbatched_matrices)
    else:
        layer_area = height * width
        layer_volume = depth

        feat_matx = tf.reshape(layers, [batch, layer_area, depth])
        feat_matx_t = tf.transpose(feat_matx, perm=[0, 2, 1])

        return tf.matmul(feat_matx, feat_matx_t) / tf.to_float(layer_volume)

def batched_layer_style_loss(args, generated_layer, style_gram_matrices):
    gram_matxs = batched_gram_matx(args, generated_layer)
    n_style_images = int(style_gram_matrices.shape[0])

    layer_sublosses = []

    # Broadcast each stacked style matrix in style_gram_matrices to every
    # element in the generated batch
    unbatched_style_matrices = tf.unstack(style_gram_matrices, axis=0, num=n_style_images)
    for style_matrix in unbatched_style_matrices:
        style_matrix = tf.expand_dims(style_matrix, 0)
        l = tf.norm(gram_matxs - style_matrix, 'fro', axis=(1,2))
        layer_sublosses.append(tf.square(l))

    return tf.reduce_sum(layer_sublosses)

# Extracts the content loss by computing squared-error loss between activations on one layer.
# Batched layers should be passed into this.
def content_loss(original_layer, generated_layer):
    base_shape = tf.shape(original_layer)
    layer_volume = tf.to_float(base_shape[1] * base_shape[2] * base_shape[3])

    return tf.reduce_sum(tf.square(generated_layer - original_layer), axis=[1,2,3]) / layer_volume

def add_loss_hyperparameters(parser):
    parser.add_argument('--enable-transpose-style-loss', action='store_true', help='If set, uses matrix-by-transpose multiplications to calculate style loss.')
