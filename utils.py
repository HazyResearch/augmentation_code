import copy
import numpy as np

import torch
import torch.nn.functional as F

from models import combine_transformed_dimension, split_transformed_dimension

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_valid_datasets(dataset,
                             valid_size=0.1,
                             random_seed=None,
                             shuffle=True):
    """
    Utility function for loading and returning train and validation
    datasets.
    Parameters:
    ------
    - dataset: the dataset, need to have train_data and train_labels attributes.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - random_seed: fix seed for reproducibility.
    - shuffle: whether to shuffle the train/validation indices.
    Returns:
    -------
    - train_dataset: training set.
    - valid_dataset: validation set.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    num_train = len(dataset)
    indices = list(range(num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset, valid_dataset = copy.copy(dataset), copy.copy(dataset)
    train_dataset.train_data = train_dataset.train_data[train_idx]
    train_dataset.train_labels = train_dataset.train_labels[train_idx]
    valid_dataset.train_data = valid_dataset.train_data[valid_idx]
    valid_dataset.train_labels = valid_dataset.train_labels[valid_idx]
    return train_dataset, valid_dataset


def train(data_loader, model, optimizer):
    model.train()
    train_loss, train_acc = [], []
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = model.predict(output)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        acc = (pred == target).sum().item() / target.size(0)
        train_loss.append(loss.item())
        train_acc.append(acc)
    return train_loss, train_acc


def train_models_compute_agreement(data_loader, models, optimizers):
    train_agreement = []
    for model in models:
        model.train()
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        pred, loss = [], []
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            output = model(data)
            pred.append(model.predict(output))
            loss_minibatch = model.loss(output, target)
            loss_minibatch.backward()
            optimizer.step()
            loss.append(loss_minibatch.item())
            # To avoid out-of-memory error, as these attributes prevent the memory from being freed
            if hasattr(model, '_avg_features'):
                del model._avg_features
            if hasattr(model, '_centered_features'):
                del model._centered_features
        loss = np.array(loss)
        pred = np.array([p.cpu().numpy() for p in pred])
        train_agreement.append((pred == pred[0]).mean(axis=1))
    return train_agreement


def train_all_epochs(train_loader,
                     valid_loader,
                     model,
                     optimizer,
                     n_epochs,
                     verbose=True):
    model.train()
    train_loss, train_acc, valid_acc = [], [], []
    for epoch in range(n_epochs):
        if verbose:
            print(f'Train Epoch: {epoch}')
        loss, acc = train(train_loader, model, optimizer)
        train_loss += loss
        train_acc += acc
        correct, total = accuracy(valid_loader, model)
        valid_acc.append(correct / total)
        if verbose:
            print(
                f'Validation set: Accuracy: {correct}/{total} ({correct/total*100:.4f}%)'
            )
    return train_loss, train_acc, valid_acc


def accuracy(data_loader, model):
    """Accuracy over all mini-batches.
    """
    training = model.training
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = model.predict(output)
            correct += (pred == target).sum().item()
            total += target.size(0)
    model.train(training)
    return correct, total


def all_losses(data_loader, model):
    """All losses over all mini-batches.
    """
    training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            losses.append([l.item() for l in model.all_losses(data, target)])
    model.train(training)
    return np.array(losses)


def agreement_kl_accuracy(data_loader, models):
    training = [model.training for model in models]
    for model in models:
        model.eval()
    valid_agreement, valid_acc, valid_kl = [], [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            pred, out = [], []
            for model in models:
                output = model(data).detach()
                out.append(output)
                pred.append(model.predict(output))
            pred = torch.stack(pred)
            out = torch.stack(out)
            log_prob = F.log_softmax(out, dim=-1)
            prob = F.softmax(out[0], dim=-1)
            valid_kl.append([F.kl_div(lp, prob, size_average=False).item() / prob.size(0) for lp in log_prob])
            valid_acc.append((pred == target).float().mean(dim=1).cpu().numpy())
            valid_agreement.append((pred == pred[0]).float().mean(dim=1).cpu().numpy())
    for model, training_ in zip(models, training):
        model.train(training_)
    return valid_agreement, valid_kl, valid_acc


def kernel_target_alignment(data_loader, model, n_passes_through_data=10):
    """Compute kernel target alignment approximately by summing over
    mini-batches. The number of mini-batches is controlled by @n_passes_through_data.
    Larger number of passes yields more accurate result, but takes longer.
    """
    inclass_kernel, kernel_fro_norm, inclass_num = [], [], []
    with torch.no_grad():
        for _ in range(n_passes_through_data):
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                features = model.features(data)
                target = target[:, None]
                same_labels = target == target.t()
                K = features @ features.t()
                inclass_kernel.append(K[same_labels].sum().item())
                kernel_fro_norm.append((K * K).sum().item())
                inclass_num.append(same_labels.long().sum().item())
    inclass_kernel = np.array(inclass_kernel)
    kernel_fro_norm = np.array(kernel_fro_norm)
    inclass_num = np.array(inclass_num)
    return inclass_kernel.mean(axis=0) / np.sqrt(kernel_fro_norm.mean(axis=0) * inclass_num.mean())


def kernel_target_alignment_augmented(data_loader, model, n_passes_through_data=10):
    """Compute kernel target alignment on augmented dataset, of the original
    features and averaged features. Alignment is approximately by summing over
    minibatches. The number of minibatches is controlled by
    @n_passes_through_data. Larger number of passes yields more accurate
    result.
    """
    inclass_kernel, kernel_fro_norm, inclass_num = [], [], []
    with torch.no_grad():
        for _ in range(n_passes_through_data):
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                n_transforms = data.size(1)
                data = combine_transformed_dimension(data)
                features = model.features(data)
                features = split_transformed_dimension(features, n_transforms)
                features_avg = features.mean(dim=1)
                features_og = features[:, 0]
                target = target[:, None]
                same_labels = target == target.t()
                K_avg = features_avg @ features_avg.t()
                K_og = features_og @ features_og.t()
                inclass_kernel.append([K_avg[same_labels].sum().item(), K_og[same_labels].sum().item()])
                kernel_fro_norm.append([(K_avg * K_avg).sum().item(), (K_og * K_og).sum().item()])
                inclass_num.append(same_labels.long().sum().item())
    inclass_kernel = np.array(inclass_kernel)
    kernel_fro_norm = np.array(kernel_fro_norm)
    inclass_num = np.array(inclass_num)
    return tuple(inclass_kernel.mean(axis=0) / np.sqrt(kernel_fro_norm.mean(axis=0) * inclass_num.mean()))
