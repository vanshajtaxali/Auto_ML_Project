import os
import argparse
import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from hyperopt import fmin, tpe, hp, space_eval, Trials
from cnn import torchModel
from datasets import K49
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main(model_config,
         data_dir='../data',
         num_epochs=10,
         batch_size=96,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    train_dataset = K49(data_dir, True, data_augmentations)
    test_dataset = K49(data_dir, False, data_augmentations)

    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    model = torchModel(model_config,
                       input_shape=(
                           train_dataset.channels,
                           train_dataset.img_rows,
                           train_dataset.img_cols
                       ),
                       num_classes=train_dataset.n_classes
                       ).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # instantiate optimizer
    optimizer = model_optimizer(model.parameters(),
                                lr=learning_rate)
    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    #logging.info('Generated Network:')
    #summary(model, (train_dataset.channels,
                   # train_dataset.img_rows,
                    #train_dataset.img_cols),
            #device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
       # logging.info('#' * 50)
       # logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = model.train_fn(optimizer, train_criterion,
                                                 train_loader, device)
       # logging.info('Train accuracy %f', train_score)

        test_score, t_loss = model.eval_fn(test_loader, train_criterion, device)
       # logging.info('Test accuracy %f', test_score)

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)


    return t_loss, test_score


def objective(x):
    # architecture parametrization
    architecture = {
        'n_layers': 1,
    }
    x, y = main(architecture)
    print("Test_score=", y)
    print("Test_loss=", x)
    return x


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """
    print('Welcome to Auto_ML project Solution...')

    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'mse': torch.nn.MSELoss}
    opti_dict = {'adam': torch.optim.Adam,
                 'adad': torch.optim.Adadelta,
                 'sgd': torch.optim.SGD}
    trials = Trials()
    best = fmin(objective,
                space={'num epochs': hp.randint('num_epochs', 20),
                       'batch_size': hp.uniform('batch_size', 140, 180),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1))
                       'train_criterion': hp.choice('train_criterion', loss_dict)
                       },
                algo=tpe.suggest,
                max_evals=10, trials=trials)
    print('best: ')
    print (best)
