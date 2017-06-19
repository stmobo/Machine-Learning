def leaky_relu(tensor_in):
    return tf.maximum(tensor_in, tensor_in * 0.01)

def debug_print(line):
    print(line)
    sys.stdout.flush()

def conv_output_size(dim_in, stride):
    if isinstance(dim_in, tf.Dimension):
        if dim_in.value is None:
            raise ValueError("Dimension size unknown!")
        dim_in = dim_in.value
    return int(math.ceil(float(dim_in) / float(stride)))

def leaky_relu(tensor_in):
    return tf.maximum(tensor_in, tensor_in * 0.01)
