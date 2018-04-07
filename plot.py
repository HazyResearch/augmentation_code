import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

sgd_n_epochs = 15
n_trials = 10


def plot_objective_difference():
    """Plot objective difference during training
    """
    for model_name in ['kernel', 'lenet']:
        for transform_name in ['rotation', 'crop', 'blur', 'rotation_crop_blur']:
            losses = np.array([np.load(f'saved/all_losses_{model_name}_{transform_name}_{seed}.npy') for seed in range(n_trials)])
            diff_og = losses[:, 1] - losses[:, 0]
            diff_1st = losses[:, 2] - losses[:, 0]
            diff_2nd_no_1st = losses[:, 3] - losses[:, 0]
            diff_2nd = losses[:, 4] - losses[:, 0]
            plt.clf()
            plt.errorbar(range(sgd_n_epochs + 1), diff_og.mean(axis=0), diff_og.std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(sgd_n_epochs + 1), diff_1st.mean(axis=0), diff_1st.std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(sgd_n_epochs + 1), diff_2nd_no_1st.mean(axis=0), diff_2nd_no_1st.std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(sgd_n_epochs + 1), diff_2nd.mean(axis=0), diff_2nd.std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.xlabel('Epoch')
            plt.ylabel('Difference in objective')
            plt.legend()
            plt.axhline(color='k')
            plt.savefig(f'figs/objective_difference_{model_name}_{transform_name}.pdf', bbox_inches='tight')

def plot_agreement_kl():
    """Plot training/valid agreements and KL divergence
    """

    for model_name in ['kernel', 'lenet']:
        for transform_name in ['rotation', 'crop', 'blur', 'rotation_crop_blur']:
            saved_arrays = [np.load(f'saved/train_valid_agreement_kl_{model_name}_{transform_name}_{seed}.npz')
                            for seed in range(n_trials)]
            train_agreement = np.array([saved['train_agreement'] for saved in saved_arrays])
            valid_agreement = np.array([saved['valid_agreement'] for saved in saved_arrays])
            valid_kl = np.array([saved['valid_kl'] for saved in saved_arrays])
            valid_acc = np.array([saved['valid_acc'] for saved in saved_arrays])

            plt.clf()
            plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 1].mean(axis=0), train_agreement[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 2].mean(axis=0), train_agreement[:, 2].std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 3].mean(axis=0), train_agreement[:, 3].std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 4].mean(axis=0), train_agreement[:, 4].std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.xlabel('Epoch')
            plt.ylabel('Prediction agreement')
            plt.legend()
            # plt.axhline(color='k')
            plt.savefig(f'figs/prediction_agreement_training_{model_name}_{transform_name}.pdf', bbox_inches='tight')

            plt.clf()
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 1].mean(axis=0), valid_agreement[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 2].mean(axis=0), valid_agreement[:, 2].std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 3].mean(axis=0), valid_agreement[:, 3].std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 4].mean(axis=0), valid_agreement[:, 4].std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.xlabel('Epoch')
            plt.ylabel('Prediction agreement')
            plt.legend()
            # plt.axhline(color='k')
            plt.savefig(f'figs/prediction_agreement_valid_{model_name}_{transform_name}.pdf', bbox_inches='tight')

            plt.clf()
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 1].mean(axis=0), valid_kl[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 2].mean(axis=0), valid_kl[:, 2].std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 3].mean(axis=0), valid_kl[:, 3].std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 4].mean(axis=0), valid_kl[:, 4].std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.xlabel('Epoch')
            plt.ylabel('Prediction KL')
            plt.legend()
            plt.axhline(color='k')
            plt.savefig(f'figs/kl_valid_{model_name}_{transform_name}.pdf', bbox_inches='tight')

            plt.clf()
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 1].mean(axis=0), valid_acc[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 2].mean(axis=0), valid_acc[:, 2].std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 3].mean(axis=0), valid_acc[:, 3].std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 4].mean(axis=0), valid_acc[:, 4].std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 0].mean(axis=0), valid_acc[:, 0].std(axis=0), fmt='o-', capsize=5, label='Exact (augmented images)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            # plt.axhline(color='k')
            plt.savefig(f'figs/accuracy_valid_{model_name}_{transform_name}.pdf', bbox_inches='tight')

            plt.clf()
            plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 1] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 1] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Original image')
            plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 2] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 2] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 3] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 3] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='2nd-order w/o 1st-order')
            plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 4] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 4] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='2nd-order')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy difference')
            plt.legend()
            plt.axhline(color='k')
            plt.savefig(f'figs/accuracy_difference_valid_{model_name}_{transform_name}.pdf', bbox_inches='tight')

def plot_agreement_kl_avg_at_layers():
    """Plot generalization difference when doing feature averaging at different layers
    """
    model_name = 'lenet'
    transform_name = 'rotation'
    saved_arrays = [np.load(f'saved/train_valid_agreement_kl_{model_name}_{transform_name}_{seed}.npz')
                    for seed in range(n_trials)]
    train_agreement = np.array([saved['train_agreement'] for saved in saved_arrays])
    valid_agreement = np.array([saved['valid_agreement'] for saved in saved_arrays])
    valid_kl = np.array([saved['valid_kl'] for saved in saved_arrays])
    valid_acc = np.array([saved['valid_acc'] for saved in saved_arrays])
    plt.clf()
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 1].mean(axis=0), train_agreement[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 4].mean(axis=0), train_agreement[:, 4].std(axis=0), fmt='o-', capsize=5, label='Averaged at 4th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 5].mean(axis=0), train_agreement[:, 5].std(axis=0), fmt='o-', capsize=5, label='Averaged at 3rd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 6].mean(axis=0), train_agreement[:, 6].std(axis=0), fmt='o-', capsize=5, label='Averaged at 2nd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 7].mean(axis=0), train_agreement[:, 7].std(axis=0), fmt='o-', capsize=5, label='Averaged at 1st layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), train_agreement[:, 8].mean(axis=0), train_agreement[:, 8].std(axis=0), fmt='o-', capsize=5, label='Averaged at 0th layer')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction agreement')
    plt.legend()
    # plt.axhline(color='k')
    plt.savefig(f'figs/prediction_agreement_training_{model_name}_{transform_name}_layers.pdf', bbox_inches='tight')

    plt.clf()
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 1].mean(axis=0), valid_agreement[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 4].mean(axis=0), valid_agreement[:, 4].std(axis=0), fmt='o-', capsize=5, label='Averaged at 4th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 5].mean(axis=0), valid_agreement[:, 5].std(axis=0), fmt='o-', capsize=5, label='Averaged at 3rd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 6].mean(axis=0), valid_agreement[:, 6].std(axis=0), fmt='o-', capsize=5, label='Averaged at 2nd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 7].mean(axis=0), valid_agreement[:, 7].std(axis=0), fmt='o-', capsize=5, label='Averaged at 1st layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_agreement[:, 8].mean(axis=0), valid_agreement[:, 8].std(axis=0), fmt='o-', capsize=5, label='Averaged at 0th layer')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction agreement')
    plt.legend()
    # plt.axhline(color='k')
    plt.savefig(f'figs/prediction_agreement_valid_{model_name}_{transform_name}_layers.pdf', bbox_inches='tight')
    plt.clf()
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 1].mean(axis=0), valid_kl[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 4].mean(axis=0), valid_kl[:, 4].std(axis=0), fmt='o-', capsize=5, label='Averaged at 4th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 5].mean(axis=0), valid_kl[:, 5].std(axis=0), fmt='o-', capsize=5, label='Averaged at 3rd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 6].mean(axis=0), valid_kl[:, 6].std(axis=0), fmt='o-', capsize=5, label='Averaged at 2nd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 7].mean(axis=0), valid_kl[:, 7].std(axis=0), fmt='o-', capsize=5, label='Averaged at 1st layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_kl[:, 8].mean(axis=0), valid_kl[:, 8].std(axis=0), fmt='o-', capsize=5, label='Averaged at 0th layer')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction KL')
    plt.legend()
    plt.axhline(color='k')
    plt.savefig(f'figs/kl_valid_{model_name}_{transform_name}_layers.pdf', bbox_inches='tight')
    plt.clf()
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 1].mean(axis=0), valid_acc[:, 1].std(axis=0), fmt='o-', capsize=5, label='Original image')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 4].mean(axis=0), valid_acc[:, 4].std(axis=0), fmt='o-', capsize=5, label='Averaged at 4th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 5].mean(axis=0), valid_acc[:, 5].std(axis=0), fmt='o-', capsize=5, label='Averaged at 3rd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 6].mean(axis=0), valid_acc[:, 6].std(axis=0), fmt='o-', capsize=5, label='Averaged at 2nd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 7].mean(axis=0), valid_acc[:, 7].std(axis=0), fmt='o-', capsize=5, label='Averaged at 1st layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 8].mean(axis=0), valid_acc[:, 8].std(axis=0), fmt='o-', capsize=5, label='Averaged at 0th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), valid_acc[:, 0].mean(axis=0), valid_acc[:, 0].std(axis=0), fmt='o-', capsize=5, label='Exact (augmented images)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.axhline(color='k')
    plt.savefig(f'figs/accuracy_valid_{model_name}_{transform_name}_layers.pdf', bbox_inches='tight')
    plt.clf()
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 1] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 1] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Original image')
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 4] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 4] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Averaged at 4th layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 5] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 5] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Averaged at 3rd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 6] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 6] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Averaged at 2nd layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 7] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 7] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Averaged at 1st layer')
    plt.errorbar(range(1, sgd_n_epochs + 1), (valid_acc[:, 8] - valid_acc[:, 0]).mean(axis=0), (valid_acc[:, 8] - valid_acc[:, 0]).std(axis=0), fmt='o-', capsize=5, label='Averaged at 0th layer')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy difference')
    plt.legend()
    plt.axhline(color='k')
    plt.savefig(f'figs/accuracy_difference_valid_{model_name}_{transform_name}_layers.pdf', bbox_inches='tight')


def plot_accuracy_vs_computation():
    """Plot computational savings when doing averaging at earlier layers of LeNet
    """
    layers = ['conv1_maxpool1', 'conv2_maxpool2', 'fc1', 'fc2', 'fc3']
    # flops = np.array([50 * 24 * 24 * 6 + 4 * 12 * 12 * 6, 24 * 8 * 8 * 16 + 4 * 4 * 4 * 16, 256 * 120 + 120, 120 * 84 + 84, 84 * 10])
    # computation_time = np.array([193, 120, 42, 42, 31])
    # offset = 3
    # computation_time -= offset
    computation_time = np.array([123, 94, 41, 40, 30])  # Measured with iPython's %timeit
    ratio = computation_time / computation_time.sum()
    n_transforms = 16
    exact = n_transforms
    avg = np.empty(6)
    avg[5] = 1.0
    avg[4] = (ratio[:4].sum() * n_transforms + ratio[4:].sum()) / exact
    avg[3] = (ratio[:3].sum() * n_transforms + ratio[3:].sum()) / exact
    avg[2] = (ratio[:2].sum() * n_transforms + ratio[2:].sum()) / exact
    avg[1] = (ratio[:1].sum() * n_transforms + ratio[1:].sum()) / exact
    avg[0] = (ratio[:0].sum() * n_transforms + ratio[0:].sum()) / exact
    model_name = 'lenet'
    transform_name = 'rotation'
    saved_arrays = [np.load(f'saved/train_valid_agreement_kl_{model_name}_{transform_name}_{seed}.npz')
                    for seed in range(n_trials)]
    valid_acc = np.array([saved['valid_acc'] for saved in saved_arrays])
    plt.clf()
    plt.errorbar(avg, valid_acc[:, [8, 7, 6, 5, 4, 0], -1].mean(axis=0), valid_acc[:, [8, 7, 6, 5, 4, 0], -1].std(axis=0), fmt='o-', capsize=5)
    plt.ylabel('Accuracy')
    plt.xlabel('Computation fraction')
    plt.savefig(f'figs/accuracy_vs_computation_{model_name}_{transform_name}.pdf', bbox_inches='tight')
    # Plot relative accuracy gain
    l, u = valid_acc[:, 8, -1].mean(axis=0), valid_acc[:, 0, -1].mean(axis=0)
    plt.figure()
    plt.errorbar(avg, (valid_acc[:, [8, 7, 6, 5, 4, 0], -1].mean(axis=0) - l) / (u - l), valid_acc[:, [8, 7, 6, 5, 4, 0], -1].std(axis=0) / (u - l), fmt='o-', capsize=10, markersize=10, linewidth=2)
    plt.ylabel('Relative accuracy gain', fontsize=16)
    plt.xlabel('Computation fraction', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'figs/accuracy_vs_computation_relative_{model_name}_{transform_name}.pdf', bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_kernel_alignment():
    """Scatter plot of accuracy vs kernel target alignment
    """
    valid_acc = []
    for model_name in ['kernel', 'lenet']:
        valid_acc_per_model = []
        # Accuracy on no transform
        saved_arrays = [np.load(f'saved/train_valid_agreement_kl_{model_name}_blur_{seed}.npz')
                        for seed in range(n_trials)]
        valid_acc_per_model.append(np.array([saved['valid_acc'] for saved in saved_arrays])[:, 1, -1])
        for transform_name in ['rotation', 'crop', 'blur', 'rotation_crop_blur', 'hflip', 'hflip_vflip', 'brightness', 'contrast']:
            saved_arrays = [np.load(f'saved/train_valid_acc_{model_name}_{transform_name}_{seed}.npz')
                            for seed in range(n_trials)]
            valid_acc_per_model.append(np.array([saved['valid_acc'] for saved in saved_arrays])[:, -1])
            # print(valid_acc.mean(axis=0)[-1], valid_acc.std(axis=0)[-1])
        valid_acc.append(valid_acc_per_model)
    valid_acc = np.array(valid_acc)
    kernel_alignment = np.load('saved/kernel_alignment.npy')

    plt.clf()
    plt.errorbar(kernel_alignment, valid_acc[0].mean(axis=-1), valid_acc[0].std(axis=-1), fmt='o', capsize=5)
    plt.axhline(valid_acc[0, 0].mean(axis=-1), color='k')
    plt.axvline(kernel_alignment[0], color='k')
    plt.errorbar(kernel_alignment[0], valid_acc[0, 0].mean(axis=-1), valid_acc[0, 0].std(axis=-1), fmt='o', capsize=5)
    plt.ylabel('Accuracy')
    plt.xlabel('Kernel target alignment')
    plt.savefig(f'figs/accuracy_vs_alignment_kernel.pdf', bbox_inches='tight')
    plt.clf()
    plt.errorbar(kernel_alignment, valid_acc[1].mean(axis=-1), valid_acc[1].std(axis=-1), fmt='o', capsize=5)
    plt.axhline(valid_acc[1, 0].mean(axis=-1), color='k')
    plt.axvline(kernel_alignment[0], color='k')
    plt.errorbar(kernel_alignment[0], valid_acc[1, 0].mean(axis=-1), valid_acc[1, 0].std(axis=-1), fmt='o', capsize=5)
    plt.ylabel('Accuracy')
    plt.xlabel('Kernel target alignment')
    plt.savefig(f'figs/accuracy_vs_alignment_lenet.pdf', bbox_inches='tight')

    plt.clf()
    sns.set_style('white')
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    ax.errorbar(kernel_alignment[0], valid_acc[0, 0].mean(axis=-1), valid_acc[0, 0].std(axis=-1), fmt='x', color='r', capsize=5)
    ax.errorbar(kernel_alignment[1], valid_acc[0, 1].mean(axis=-1), valid_acc[0, 1].std(axis=-1), fmt='s', color='b', capsize=5)
    ax.errorbar(kernel_alignment[2], valid_acc[0, 2].mean(axis=-1), valid_acc[0, 2].std(axis=-1), fmt='s', color='g', capsize=5)
    ax.errorbar(kernel_alignment[3], valid_acc[0, 3].mean(axis=-1), valid_acc[0, 3].std(axis=-1), fmt='o', color='b', capsize=5)
    ax.errorbar(kernel_alignment[4], valid_acc[0, 4].mean(axis=-1), valid_acc[0, 4].std(axis=-1), fmt='s', color='tab:orange', capsize=5)
    ax.errorbar(kernel_alignment[5], valid_acc[0, 5].mean(axis=-1), valid_acc[0, 5].std(axis=-1), fmt='D', color='g', capsize=5)
    ax.errorbar(kernel_alignment[6], valid_acc[0, 6].mean(axis=-1), valid_acc[0, 6].std(axis=-1), fmt='o', color='g', capsize=5)
    ax.errorbar(kernel_alignment[7], valid_acc[0, 7].mean(axis=-1), valid_acc[0, 7].std(axis=-1), fmt='D', color='b', capsize=5)
    ax.errorbar(kernel_alignment[8], valid_acc[0, 8].mean(axis=-1), valid_acc[0, 8].std(axis=-1), fmt='D', color='m', capsize=5)
    ax.axhline(valid_acc[0, 0].mean(axis=-1), color='k')
    ax.axvline(kernel_alignment[0], color='k')
    ax.set_yticks([0.94, 0.96, 0.98])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('RBF Kernel', fontsize=16)
    ax = plt.subplot(1, 2, 2)
    ax.errorbar(kernel_alignment[0], valid_acc[1, 0].mean(axis=-1), valid_acc[1, 0].std(axis=-1), fmt='x', color='r', capsize=5, label='original')
    ax.errorbar(kernel_alignment[1], valid_acc[1, 1].mean(axis=-1), valid_acc[1, 1].std(axis=-1), fmt='s', color='b', capsize=5, label='rotation')
    ax.errorbar(kernel_alignment[2], valid_acc[1, 2].mean(axis=-1), valid_acc[1, 2].std(axis=-1), fmt='s', color='g', capsize=5, label='crop')
    ax.errorbar(kernel_alignment[3], valid_acc[1, 3].mean(axis=-1), valid_acc[1, 3].std(axis=-1), fmt='o', color='b', capsize=5, label='blur')
    ax.errorbar(kernel_alignment[4], valid_acc[1, 4].mean(axis=-1), valid_acc[1, 4].std(axis=-1), fmt='s', color='tab:orange', capsize=5, label='rotation, crop, blur')
    ax.errorbar(kernel_alignment[5], valid_acc[1, 5].mean(axis=-1), valid_acc[1, 5].std(axis=-1), fmt='D', color='g', capsize=5, label='h. flip')
    ax.errorbar(kernel_alignment[6], valid_acc[1, 6].mean(axis=-1), valid_acc[1, 6].std(axis=-1), fmt='o', color='g', capsize=5, label='h. flip, v. flip')
    ax.errorbar(kernel_alignment[7], valid_acc[1, 7].mean(axis=-1), valid_acc[1, 7].std(axis=-1), fmt='D', color='b', capsize=5, label='brightness')
    ax.errorbar(kernel_alignment[8], valid_acc[1, 8].mean(axis=-1), valid_acc[1, 8].std(axis=-1), fmt='D', color='m', capsize=5, label='contrast')
    ax.axhline(valid_acc[1, 0].mean(axis=-1), color='k')
    ax.axvline(kernel_alignment[0], color='k')
    ax.set_yticks([0.97, 0.98, 0.99])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('LeNet', fontsize=16)
    # sns.despine()
    # labels = ['original', 'rotation', 'crop', 'blur', 'rot., crop, blur', 'h. flip', 'h. flip, v. flip', 'brightness', 'contrast']
    # plt.legend(labels, loc='upper center', bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0,0,1,1), ncol=3, fontsize=14)
    plt.legend(loc='upper center', bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0,0.07,1,1), ncol=3, fontsize=16, frameon=True, edgecolor='k')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, top=0.75, bottom=0.1)
    plt.suptitle('Kernel target alignment', x=0.5, y=0.05, fontsize=16)
    # ax.set_ylabel('Accuracy')
    # plt.xlabel('Kernel target alignment')
    plt.savefig(f'figs/accuracy_vs_alignment.pdf', bbox_inches='tight')

def main():
    pathlib.Path('figs').mkdir(parents=True, exist_ok=True)
    plot_objective_difference()
    plot_agreement_kl()
    plot_agreement_kl_avg_at_layers()
    plot_accuracy_vs_computation()
    plot_accuracy_vs_kernel_alignment()

if __name__ == '__main__':
    main()
