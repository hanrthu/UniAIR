import torch
from torch_scatter import scatter_add

def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Example::

        >>> mask = multi_slice_mask(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([1, 1, 1, 0, 1, 1])).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    if slices.numel():
        assert slices.min() >= 0 and slices.max() <= length
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask

def padded_to_variadic(padded, size):
    """
    Convert a padded tensor to a variadic tensor.

    Parameters:
        padded (Tensor): padded tensor of shape :math:`(N, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    num_sample, max_size = padded.shape[:2]
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    return padded[mask]

def variadic_to_padded(input, size, value=0):
    """
    Convert a variadic tensor to a padded tensor.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        value (scalar): fill value for padding

    Returns:
        (Tensor, BoolTensor): padded tensor and mask
    """
    num_sample = len(size)
    max_size = size.max()
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    shape = (num_sample, max_size) + input.shape[1:]
    padded = torch.full(shape, value, dtype=input.dtype, device=size.device)
    padded[mask] = input
    return padded, mask

def variadic_arange(size):
    """
    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range

def variadic_meshgrid(input1, size1, input2, size2):
    """
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]
