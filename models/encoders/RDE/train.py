import torch
import numpy as np
from data.protein.residue_constants import chi_pi_periodic, lexico_to_type_id
import matplotlib.pyplot as plt
def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def aggregate_sidechain_accuracy(aa, chi_pred, chi_native, chi_mask):
    aa = aa.reshape(-1)
    chi_mask = chi_mask.reshape(-1, 4)
    diff = torch.min(
        (chi_pred - chi_native) % (2 * np.pi),
        (chi_native - chi_pred) % (2 * np.pi),
    )  # (N, L, 4)
    diff = torch.rad2deg(diff)
    diff = diff.reshape(-1, 4)

    diff_flip = torch.min(
        ((chi_pred + np.pi) - chi_native) % (2 * np.pi),
        (chi_native - (chi_pred + np.pi)) % (2 * np.pi),
    )
    diff_flip = torch.rad2deg(diff_flip)
    diff_flip = diff_flip.reshape(-1, 4)

    acc = [{j: [] for j in range(1, 4 + 1)} for i in range(20)]
    for i in range(aa.size(0)):
        for j in range(4):
            chi_number = j + 1
            if not chi_mask[i, j].item(): continue
            # TODO: Needs to be double_check
            if chi_pi_periodic[lexico_to_type_id[aa[i].item()]][chi_number - 1]:
                diff_this = min(diff[i, j].item(), diff_flip[i, j].item())
            else:
                diff_this = diff[i, j].item()
            acc[aa[i].item()][chi_number].append(diff_this)

    table = np.full((20, 4), np.nan)
    for i in range(20):
        for j in range(1, 4 + 1):
            if len(acc[i][j]) > 0:
                table[i, j - 1] = np.mean(acc[i][j])
    return table


def make_sidechain_accuracy_table_image(tag: str, diff: np.ndarray):
    from Bio.PDB.Polypeptide import index_to_three
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    rows = [index_to_three(i) for i in range(20)]
    cell_text = diff.tolist()
    fig, ax = plt.subplots(dpi=200)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(tag)
    ax.table(
        cellText=cell_text,
        colLabels=columns,
        rowLabels=rows,
        loc='center'
    )
    return fig

