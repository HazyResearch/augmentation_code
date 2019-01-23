import copy
import numpy as np
import pathlib

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import LinearLogisticRegression, RBFLogisticRegression, RBFLogisticRegressionRotated, LinearLogisticRegressionAug, RBFLogisticRegressionAug, LeNet, LeNetAug, combine_transformed_dimension, split_transformed_dimension
from augmentation import copy_with_new_transform, augment_transforms, rotation, resized_crop, blur, rotation_crop_blur, hflip, hflip_vflip, brightness, contrast
from utils import get_train_valid_datasets, train, train_all_epochs, accuracy, all_losses, train_models_compute_agreement, agreement_kl_accuracy, kernel_target_alignment, kernel_target_alignment_augmented, kernel_target_alignment_augmented_no_avg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir = 'saved_cifar10'
# save_dir = 'saved_cifar10_rerun_50_epochs'
# save_dir = 'saved_cifar10_rerun_100_epochs'
# save_dir = 'saved_cifar10_basic_models_3_channels'

batch_size = 256
if device.type == 'cuda':
    loader_args = {'num_workers': 32, 'pin_memory': True}
    # loader_args = {'num_workers': 16, 'pin_memory': True}
else:
    loader_args = {'num_workers': 4, 'pin_memory': False}


def loader_from_dataset(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, **loader_args)

# Construct loader from CIFAR10 dataset, then construct loaders corresponding to
# augmented dataset (wrt to different transformations).
cifar10_normalize = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
cifar10_train = datasets.CIFAR10(
    '../data', train=True, download=True, transform=cifar10_normalize)
cifar10_test = datasets.CIFAR10(
    '../data', train=False, download=True, transform=cifar10_normalize)
# For some reason the train labels are lists instead of torch.LongTensor
cifar10_train.train_labels = torch.LongTensor(cifar10_train.train_labels)
cifar10_test.test_labels = torch.LongTensor(cifar10_test.test_labels)
cifar10_train, cifar10_valid = get_train_valid_datasets(cifar10_train)
train_loader = loader_from_dataset(cifar10_train)
valid_loader = loader_from_dataset(cifar10_valid)
test_loader = loader_from_dataset(cifar10_test)

cifar10_normalize_rotate = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation((-5, 5)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
cifar10_test_rotated = datasets.CIFAR10(
    '../data', train=False, download=True, transform=cifar10_normalize_rotate)
test_loader_rotated = loader_from_dataset(cifar10_test_rotated)

augmentations = [rotation(cifar10_train, cifar10_normalize, angles=range(-5, 6, 1)),
                 resized_crop(cifar10_train, cifar10_normalize, size=32),
                 blur(cifar10_train, cifar10_normalize),
                 rotation_crop_blur(cifar10_train, cifar10_normalize, size=32),
                 hflip(cifar10_train, cifar10_normalize),
                 hflip_vflip(cifar10_train, cifar10_normalize),
                 brightness(cifar10_train, cifar10_normalize),
                 contrast(cifar10_train, cifar10_normalize)]

n_channels = 3
size = 32
n_features = n_channels * size * size
n_classes = 10
gamma = 0.003  # gamma hyperparam for RBF kernel exp(-gamma ||x - y||^2). Best gamma is around 0.001--0.003
n_components = 10000
sgd_n_epochs = 15
n_trials = 10

model_factories = {'linear': lambda: LinearLogisticRegressionAug(n_features, n_classes),
                   'kernel': lambda: RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, approx=False),
                   'lenet': lambda: LeNetAug(n_channels=n_channels, size=size, approx=False)}


def sgd_opt_from_model(model, learning_rate=0.1, momentum=0.9, weight_decay=0.000):
    # return optim.SGD((p for p in model.parameters() if p.requires_grad),
                     # lr=learning_rate, momentum=momentum,
                     # weight_decay=weight_decay)
    return optim.Adam((p for p in model.parameters() if p.requires_grad),
                     weight_decay=weight_decay)


def train_basic_models(train_loader, augmented_loader):
    """Train a few simple models with data augmentation / approximation, as a
    sanity check.
    """
    test_acc = []
    test_acc_rotated = []
    for seed in range(n_trials):
        print(f'Seed: {seed}')
        torch.manual_seed(seed)
        models = [
            LinearLogisticRegressionAug(n_features, n_classes),  # No augmentation, accuracy around 92.5%
            RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components), # Accuracy around 97.5%
            LeNetAug(n_channels=n_channels, size=size), # Accuracy around 98.7%
            LinearLogisticRegressionAug(n_features, n_classes, approx=False),  # Augmented data, exact objective
            RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, approx=False),
            LeNetAug(n_channels=n_channels, size=size, approx=False),
            LinearLogisticRegressionAug(n_features, n_classes, regularization=False),  # Augmented data, 1st order approx
            RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, regularization=False),
            LeNetAug(n_channels=n_channels, size=size),
            LinearLogisticRegressionAug(n_features, n_classes, feature_avg=False, regularization=True),  # Augmented data, 2nd order no 1st approx
            RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, feature_avg=False, regularization=True),
            LeNetAug(n_channels=n_channels, size=size, feature_avg=False, regularization=True),
            LinearLogisticRegressionAug(n_features, n_classes, regularization=True),  # Augmented data, 2nd order approx
            RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, regularization=True),
            LeNetAug(n_channels=n_channels, size=size, regularization=True),
            RBFLogisticRegressionRotated(n_features, n_classes, gamma=gamma, n_components=n_components, size=size, n_channels=n_channels)
                ]
        loaders = [train_loader, train_loader, train_loader,
                augmented_loader, augmented_loader, augmented_loader,
                augmented_loader, augmented_loader, augmented_loader,
                augmented_loader, augmented_loader, augmented_loader,
                augmented_loader, augmented_loader, augmented_loader,
                train_loader]
        test_acc_per_seed = []
        test_acc_rotated_per_seed = []
        import time
        for model, loader in zip(models, loaders):
            start = time.time()
            # model = RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components)
            model = RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, approx=False)
            # model = RBFLogisticRegressionRotated(n_features, n_classes, gamma=gamma, n_components=n_components, n_channels=n_channels, size=size)
            # end = time.time()
            # print(end - start)
            model.to(device)
            optimizer = sgd_opt_from_model(model)
            # start = time.time()
            train_loss, train_acc, valid_acc = train_all_epochs(loader, valid_loader, model,
                                                                optimizer, sgd_n_epochs, verbose=True)
            end = time.time()
            print(end - start)
            correct, total = accuracy(test_loader, model)
            print(f'Test set: Accuracy: {correct}/{total} ({correct/total*100:.4f}%)')
            correct_rotated, total_rotated = accuracy(test_loader_rotated, model)
            print(f'Test set rotated: Accuracy: {correct_rotated}/{total_rotated} ({correct_rotated/total_rotated*100:.4f}%)\n')
            test_acc_per_seed.append(correct / total)
            test_acc_rotated_per_seed.append(correct_rotated / total_rotated)
        np.save(f'{save_dir}/basic_models_cifar10_test_accuracy_last_{seed}.np', np.array(test_acc_per_seed))
        np.save(f'{save_dir}/basic_models_cifar10_test_accuracy_rotated_last_{seed}.np', np.array(test_acc_rotated_per_seed))
        test_acc.append(np.array(test_acc_per_seed))
        test_acc_rotated.append(np.array(test_acc_rotated_per_seed))
    test_acc = np.array(test_acc)
    test_acc_rotated = np.array(test_acc_rotated)
    np.save(f'{save_dir}/basic_models_cifar10_test_accuracy_last', test_acc)
    np.save(f'{save_dir}/basic_models_cifar10_test_accuracy_rotated_last', test_acc_rotated)


def objective_difference(augmentations):
    """Measure the difference in approximate and true objectives as we train on
    the true objective.
    """
    # for model_name in ['kernel', 'lenet']:
    for model_name in ['lenet']:
        for augmentation in augmentations:
            for seed in range(n_trials):
                print(f'Seed: {seed}')
                torch.manual_seed(seed)
                model = model_factories[model_name]().to(device)
                optimizer = sgd_opt_from_model(model)
                loader = loader_from_dataset(augmentation.dataset)
                model.train()
                losses = []
                losses.append(all_losses(loader, model).mean(axis=0))
                train_loss, train_acc, valid_acc = [], [], []
                for epoch in range(sgd_n_epochs):
                    train_loss_epoch, train_acc_epoch = train(loader, model, optimizer)
                    train_loss += train_loss_epoch
                    train_acc += train_acc_epoch
                    print(f'Train Epoch: {epoch}')
                    correct, total = accuracy(valid_loader, model)
                    valid_acc.append(correct / total)
                    print(
                        f'Validation set: Accuracy: {correct}/{total} ({correct/total*100:.4f}%)'
                    )
                    losses.append(np.array(all_losses(loader, model)).mean(axis=0))
                train_loss, train_acc, valid_acc = np.array(train_loss), np.array(train_acc), np.array(valid_acc)
                np.savez(f'{save_dir}/train_valid_acc_{model_name}_{augmentation.name}_{seed}.npz',
                        train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc)
                losses = np.array(losses).T
                np.save(f'{save_dir}/all_losses_{model_name}_{augmentation.name}_{seed}.npy', losses)


def accuracy_on_true_objective(augmentations):
    """Measure the accuracy when trained on true augmented objective.
    """
    for model_name in ['kernel', 'lenet']:
        for augmentation in augmentations:
            for seed in range(n_trials):
                print(f'Seed: {seed}')
                torch.manual_seed(seed)
                model = model_factories[model_name]().to(device)
                optimizer = sgd_opt_from_model(model)
                loader = loader_from_dataset(augmentation.dataset)
                train_loss, train_acc, valid_acc = train_all_epochs(loader, valid_loader, model, optimizer, sgd_n_epochs)
                train_loss, train_acc, valid_acc = np.array(train_loss), np.array(train_acc), np.array(valid_acc)
                np.savez(f'{save_dir}/train_valid_acc_{model_name}_{augmentation.name}_{seed}.npz',
                        train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc)


def exact_to_og_model(model):
    """Convert model training on exact augmented objective to model training on
    original data.
    """
    model_og = copy.deepcopy(model)
    model_og.approx = True
    model_og.feature_avg = False
    model_og.regularization = False
    return model_og


def exact_to_1st_order_model(model):
    """Convert model training on exact augmented objective to model training on
    1st order approximation.
    """
    model_1st = copy.deepcopy(model)
    model_1st.approx = True
    model_1st.feature_avg = True
    model_1st.regularization = False
    return model_1st


def exact_to_2nd_order_no_1st_model(model):
    """Convert model training on exact augmented objective to model training on
    2nd order approximation without feature averaging (1st order approx).
    """
    model_2nd_no_1st = copy.deepcopy(model)
    model_2nd_no_1st.approx = True
    model_2nd_no_1st.feature_avg = False
    model_2nd_no_1st.regularization = True
    return model_2nd_no_1st


def exact_to_2nd_order_model(model):
    """Convert model training on exact augmented objective to model training on
    2nd order approximation.
    """
    model_2nd = copy.deepcopy(model)
    model_2nd.approx = True
    model_2nd.feature_avg = True
    model_2nd.regularization = True
    return model_2nd


def exact_to_2nd_order_model_layer_avg(model, layer_to_avg=3):
    """Convert LeNet model training on exact augmented objective to model
    training on 2nd order approximation, but approximation is done at different
    layers.
    """
    model_2nd = copy.deepcopy(model)
    model_2nd.approx = True
    model_2nd.feature_avg = True
    model_2nd.regularization = True
    model_2nd.layer_to_avg = layer_to_avg
    # Can't use the regularization function specialized to linear model unless
    # averaging at layer 4.
    if layer_to_avg != 4:
        model.regularization_2nd_order = model.regularization_2nd_order_general
    return model_2nd


def agreement_kl_difference(augmentations):
    """Measure the agreement and KL divergence between the predictions produced
    by model trained on exact augmentation objectives vs models trained on
    approximate objectives.
    """
    model_variants = {'kernel': lambda model: [model, exact_to_og_model(model), exact_to_1st_order_model(model),
                                          exact_to_2nd_order_no_1st_model(model), exact_to_2nd_order_model(model)],
                      'lenet': lambda model: [model, exact_to_og_model(model), exact_to_1st_order_model(model),
                                          exact_to_2nd_order_no_1st_model(model)] +
                                        [exact_to_2nd_order_model_layer_avg(model, layer_to_avg) for layer_to_avg in [4, 3, 2, 1, 0]]}

    for model_name in ['kernel', 'lenet']:
    # for model_name in ['lenet']:
        for augmentation in augmentations:
            for seed in range(5, 5 + n_trials):
                print(f'Seed: {seed}')
                torch.manual_seed(n_trials + seed)
                loader = loader_from_dataset(augmentation.dataset)
                model = model_factories[model_name]()
                models = model_variants[model_name](model)
                for model in models:
                    model.to(device)
                optimizers = [sgd_opt_from_model(model) for model in models]
                for model in models:
                    model.train()
                train_agreement, valid_agreement, valid_acc, valid_kl = [], [], [], []
                for epoch in range(sgd_n_epochs):
                    print(f'Train Epoch: {epoch}')
                    train_agreement_epoch = train_models_compute_agreement(loader, models, optimizers)
                    train_agreement.append(np.array(train_agreement_epoch).mean(axis=0))
                    # Agreement and KL on validation set
                    valid_agreement_epoch, valid_kl_epoch, valid_acc_epoch = agreement_kl_accuracy(valid_loader, models)
                    valid_agreement.append(np.array(valid_agreement_epoch).mean(axis=0))
                    valid_acc.append(np.array(valid_acc_epoch).mean(axis=0))
                    valid_kl.append(np.array(valid_kl_epoch).mean(axis=0))
                train_agreement = np.array(train_agreement).T
                valid_agreement = np.array(valid_agreement).T
                valid_acc = np.array(valid_acc).T
                valid_kl = np.array(valid_kl).T
                np.savez(f'{save_dir}/train_valid_agreement_kl_{model_name}_{augmentation.name}_{seed}.npz',
                        train_agreement=train_agreement, valid_agreement=valid_agreement, valid_acc=valid_acc, valid_kl=valid_kl)


def find_gamma_by_alignment(train_loader, gamma_vals=(0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)):
    """Example use of kernel target alignment: to pick the hyperparameter gamma
    of the RBF kernel exp(-gamma ||x-y||^2) by computing the kernel target
    alignment of the random features wrt different values of gamma.
    The value of gamma giving the highest alignment is likely the best gamma.
    """
    for gamma in gamma_vals:
        model = RBFLogisticRegressionAug(n_features, n_classes, gamma=gamma, n_components=n_components, approx=False).to(device)
        print(kernel_target_alignment(train_loader, model))
    # Best gamma is 0.003


def alignment_comparison(augmentations):
    """Compute the kernel target alignment of different augmentations.
    """
    alignment = []
    model_name = 'kernel'
    for augmentation in augmentations[:1]:
        print(augmentation.name)
        loader = loader_from_dataset(augmentation.dataset)
        model = model_factories[model_name]().to(device)
        alignment.append(kernel_target_alignment_augmented(loader, model, n_passes_through_data=50))
    print(alignment)
    alignment = np.array(alignment)
    alignment_no_transform = alignment[:, 1].mean()
    np.save(f'{save_dir}/kernel_alignment.npy', np.array([alignment_no_transform] + list(alignment[:, 0])))

    alignment_no_avg = []
    model_name = 'kernel'
    for augmentation in augmentations:
        print(augmentation.name)
        loader = loader_from_dataset(augmentation.dataset)
        model = model_factories[model_name]().to(device)
        alignment_no_avg.append(kernel_target_alignment_augmented_no_avg(loader, model, n_passes_through_data=10))
    alignment_no_avg = np.array(alignment_no_avg)
    np.save(f'{save_dir}/kernel_alignment_no_avg.npy', alignment_no_avg)


def measure_computation_fraction_lenet(train_loader):
    """Measure percentage of computation time spent in each layer of LeNet.
    """
    model = LeNet(n_channels=n_channels, size=32).to(device)
    loader = train_loader
    it = iter(loader)
    data, target = next(it)
    data, target = data.to(device), target.to(device)
    # We use iPython's %timeit. Uncomment and copy these to iPython.
    # %timeit feat1 = model.layer_1(data)
    # feat1 = model.layer_1(data)
    # %timeit feat2 = model.layer_2(feat1)
    # feat2 = model.layer_2(feat1)
    # %timeit feat3 = model.layer_3(feat2)
    # feat3 = model.layer_3(feat2)
    # %timeit feat4 = model.layer_4(feat3)
    # feat4 = model.layer_4(feat3)
    # %timeit output = model.output_from_features(feat4)
    # %timeit output = model(data)



def main():
    pathlib.Path(f'{save_dir}').mkdir(parents=True, exist_ok=True)
    # train_basic_models(train_loader, loader_from_dataset(augmentations[0].dataset))
    # objective_difference(augmentations[:4])
    accuracy_on_true_objective(augmentations[:1])
    # agreement_kl_difference(augmentations[:1])
    # find_gamma_by_alignment(train_loader)
    # alignment_comparison(augmentations)
    # alignment_lenet(augmentations)

if __name__ == '__main__':
    main()
