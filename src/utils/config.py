CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar': ([16, 'MaxPooling', 32, 'MaxPooling', 'Flatten'], 3, 10, 2048, 64),
    'cifar100-c25': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 25, 128, 128),
    'cifar100-c30': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 30, 2048, 128),
    'cifar100-c50': ([32, 'MaxPooling', 64, 'MaxPooling', 128, 'Flatten'], 3, 50, 2048, 128),

    'emnist': ([6, 16, 'Flatten'], 1, 26, 784, 32),
    'mnist': ([6, 16, 'Flatten'], 1, 10, 784, 32),
    'mnist_cnn1': ([6, 'MaxPooling', 16, 'MaxPooling', 'Flatten'], 1, 10, 64, 32),
    'mnist_cnn2': ([16, 'MaxPooling', 32, 'MaxPooling', 'Flatten'], 1, 10, 128, 32),
    'celeb': ([16, 'MaxPooling', 32, 'MaxPooling', 64, 'MaxPooling', 'Flatten'], 3, 2, 64, 32)
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'cifar': (512, 32, 3, 10, 64),
    'celeb': (128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': (128, 32, 1, 10, 32),
    'mnist-cnn2': (64, 32, 1, 10, 32),
    'mnist-cnn3': (64, 32, 1, 10, 16),
    'emnist': (256, 32, 1, 26, 32),
    'emnist-cnn0': (256, 32, 1, 26, 64),
    'emnist-cnn1': (128, 32, 1, 26, 32),
    'emnist-cnn2': (128, 32, 1, 26, 16),
    'emnist-cnn3': (64, 32, 1, 26, 32),
}