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
    'celeb': ([16, 'MaxPooling', 32, 'MaxPooling', 64, 'MaxPooling', 'Flatten'], 3, 2, 64, 32),
    'batch_size': 32,
    'gen_batch_size': 32,
    'latent_layer_idx': -1,
    'gen_inference_size': 128
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    # 'cifar': (512, 32, 3, 10, 64),
    'cifar': (1028, 512, 3, 10, 64),
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



RUNCONFIGS = {
    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 26,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },

    'celeb':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },

}

