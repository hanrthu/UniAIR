import os
from torch.utils import cpp_extension
from torch import distributed as dist
import torch



class cached_property(property):
    """
    Cache the property once computed.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        result = self.func(obj)
        obj.__dict__[self.func.__name__] = result
        return result

def get_rank():
    """
    Get the rank of this process in distributed processes.

    Return 0 for single process case.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

class LazyExtensionLoader(object):

    def __init__(self, name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None,
                 extra_include_paths=None, build_directory=None, verbose=False, **kwargs):
        self.name = name
        self.sources = sources
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_ldflags = extra_ldflags
        self.extra_include_paths = extra_include_paths
        worker_name = "%s_%d" % (name, get_rank())
        self.build_directory = build_directory or cpp_extension._get_build_directory(worker_name, verbose)
        self.verbose = verbose
        self.kwargs = kwargs

    def __getattr__(self, key):
        return getattr(self.module, key)

    @cached_property
    def module(self):
        return cpp_extension.load(self.name, self.sources, self.extra_cflags, self.extra_cuda_cflags,
                                  self.extra_ldflags, self.extra_include_paths, self.build_directory,
                                  self.verbose, **self.kwargs)

def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    """
    Load a PyTorch C++ extension just-in-time (JIT).
    Automatically decide the compilation flags if not specified.

    This function performs lazy evaluation and is multi-process-safe.

    See `torch.utils.cpp_extension.load`_ for more details.

    .. _torch.utils.cpp_extension.load:
        https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
    """
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return LazyExtensionLoader(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)

def sparse_coo_tensor(indices, values, size):
    """
    Construct a sparse COO tensor without index check. Much faster than `torch.sparse_coo_tensor`_.

    .. _torch.sparse_coo_tensor:
        https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html

    Parameters:
        indices (Tensor): 2D indices of shape (2, n)
        values (Tensor): values of shape (n,)
        size (list): size of the tensor
    """
    return torch_ext.sparse_coo_tensor_unsafe(indices, values, size)

path = "/home/HR/UniPPI/extension"
torch_ext = load_extension("torch_ext", [os.path.join(path, "torch_ext.cpp")])