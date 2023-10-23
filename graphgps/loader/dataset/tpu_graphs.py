from typing import Optional, Callable, List
import copy
import re
import os
import glob
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import (Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import InMemoryDataset

# import copy
# import os.path as osp
# import warnings
# from abc import ABC
# from collections.abc import Mapping, Sequence
# from typing import (
#     Any,
#     Callable,
#     Dict,
#     Iterable,
#     List,
#     Optional,
#     Tuple,
#     Type,
#     Union,
# )

# import torch
# from torch import Tensor
# from tqdm import tqdm

# import torch_geometric
# from torch_geometric.data import Batch, Data
# from torch_geometric.data.collate import collate
# from torch_geometric.data.data import BaseData
# from torch_geometric.data.dataset import  IndexType
# from torch_geometric.data.separate import separate

# import copy
# import os.path as osp
# import re
# import sys
# import warnings
# from abc import ABC, abstractmethod
# from collections.abc import Sequence
# from typing import Any, Callable, List, Optional, Tuple, Union

# import numpy as np
# import torch.utils.data
# from torch import Tensor

# from torch_geometric.data.data import BaseData
# from torch_geometric.data.makedirs import makedirs

# IndexType = Union[slice, Tensor, np.ndarray, Sequence]


# class Dataset(torch.utils.data.Dataset, ABC):
#     r"""Dataset base class for creating graph datasets.
#     See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
#     create_dataset.html>`__ for the accompanying tutorial.

#     Args:
#         root (str, optional): Root directory where the dataset should be saved.
#             (optional: :obj:`None`)
#         transform (callable, optional): A function/transform that takes in a
#             :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             transformed version.
#             The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             a :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             transformed version.
#             The data object will be transformed before being saved to disk.
#             (default: :obj:`None`)
#         pre_filter (callable, optional): A function that takes in a
#             :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             boolean value, indicating whether the data object should be
#             included in the final dataset. (default: :obj:`None`)
#         log (bool, optional): Whether to print any console output while
#             downloading and processing the dataset. (default: :obj:`True`)
#     """
#     @property
#     def raw_file_names(self) -> Union[str, List[str], Tuple]:
#         r"""The name of the files in the :obj:`self.raw_dir` folder that must
#         be present in order to skip downloading."""
#         raise NotImplementedError

#     @property
#     def processed_file_names(self) -> Union[str, List[str], Tuple]:
#         r"""The name of the files in the :obj:`self.processed_dir` folder that
#         must be present in order to skip processing."""
#         raise NotImplementedError

#     def download(self):
#         r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
#         raise NotImplementedError

#     def process(self):
#         r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
#         raise NotImplementedError

#     @abstractmethod
#     def len(self) -> int:
#         r"""Returns the number of data objects stored in the dataset."""
#         raise NotImplementedError

#     @abstractmethod
#     def get(self, idx: int) -> BaseData:
#         r"""Gets the data object at index :obj:`idx`."""
#         raise NotImplementedError

#     def __init__(
#         self,
#         root: Optional[str] = None,
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#         pre_filter: Optional[Callable] = None,
#         log: bool = True,
#     ):
#         super().__init__()

#         if isinstance(root, str):
#             root = osp.expanduser(osp.normpath(root))

#         self.root = root
#         self.transform = transform
#         self.pre_transform = pre_transform
#         self.pre_filter = pre_filter
#         self.log = log
#         self._indices: Optional[Sequence] = None

#         if self.has_download:
#             self._download()

#         if self.has_process:
#             self._process()

#     def indices(self) -> Sequence:
#         return range(self.len()) if self._indices is None else self._indices

#     @property
#     def raw_dir(self) -> str:
#         return osp.join(self.root, 'raw')

#     @property
#     def processed_dir(self) -> str:
#         tmp = osp.join(self.root, 'processed', self.source, self.search)
#         print("processed_dir:", tmp)
#         os.makedirs(tmp, exist_ok=True)
#         return tmp

#     @property
#     def num_node_features(self) -> int:
#         r"""Returns the number of features per node in the dataset."""
#         data = self[0]
#         # Do not fill cache for `InMemoryDataset`:
#         if hasattr(self, '_data_list') and self._data_list is not None:
#             self._data_list[0] = None
#         data = data[0] if isinstance(data, tuple) else data
#         if hasattr(data, 'num_node_features'):
#             return data.num_node_features
#         raise AttributeError(f"'{data.__class__.__name__}' object has no "
#                              f"attribute 'num_node_features'")

#     @property
#     def num_features(self) -> int:
#         r"""Returns the number of features per node in the dataset.
#         Alias for :py:attr:`~num_node_features`."""
#         return self.num_node_features

#     @property
#     def num_edge_features(self) -> int:
#         r"""Returns the number of features per edge in the dataset."""
#         data = self[0]
#         # Do not fill cache for `InMemoryDataset`:
#         if hasattr(self, '_data_list') and self._data_list is not None:
#             self._data_list[0] = None
#         data = data[0] if isinstance(data, tuple) else data
#         if hasattr(data, 'num_edge_features'):
#             return data.num_edge_features
#         raise AttributeError(f"'{data.__class__.__name__}' object has no "
#                              f"attribute 'num_edge_features'")

#     def _infer_num_classes(self, y: Optional[Tensor]) -> int:
#         if y is None:
#             return 0
#         elif y.numel() == y.size(0) and not torch.is_floating_point(y):
#             return int(y.max()) + 1
#         elif y.numel() == y.size(0) and torch.is_floating_point(y):
#             return torch.unique(y).numel()
#         else:
#             return y.size(-1)

#     @property
#     def num_classes(self) -> int:
#         r"""Returns the number of classes in the dataset."""
#         # We iterate over the dataset and collect all labels to determine the
#         # maximum number of classes. Importantly, in rare cases, `__getitem__`
#         # may produce a tuple of data objects (e.g., when used in combination
#         # with `RandomLinkSplit`, so we take care of this case here as well:
#         data_list = _get_flattened_data_list([data for data in self])
#         y = torch.cat([data.y for data in data_list if 'y' in data], dim=0)

#         # Do not fill cache for `InMemoryDataset`:
#         if hasattr(self, '_data_list') and self._data_list is not None:
#             self._data_list = self.len() * [None]
#         return self._infer_num_classes(y)

#     @property
#     def raw_paths(self) -> List[str]:
#         r"""The absolute filepaths that must be present in order to skip
#         downloading."""
#         files = self.raw_file_names
#         # Prevent a common source of error in which `file_names` are not
#         # defined as a property.
#         if isinstance(files, Callable):
#             files = files()
#         return [osp.join(self.raw_dir, f) for f in to_list(files)]

#     @property
#     def processed_paths(self) -> List[str]:
#         r"""The absolute filepaths that must be present in order to skip
#         processing."""
#         files = self.processed_file_names
#         # Prevent a common source of error in which `file_names` are not
#         # defined as a property.
#         if isinstance(files, Callable):
#             files = files()
#         return [osp.join(self.processed_dir, f) for f in to_list(files)]

#     @property
#     def has_download(self) -> bool:
#         r"""Checks whether the dataset defines a :meth:`download` method."""
#         return overrides_method(self.__class__, 'download')

#     def _download(self):
#         if files_exist(self.raw_paths):  # pragma: no cover
#             return

#         makedirs(self.raw_dir)
#         self.download()

#     @property
#     def has_process(self) -> bool:
#         r"""Checks whether the dataset defines a :meth:`process` method."""
#         return overrides_method(self.__class__, 'process')

#     def _process(self):
#         f = osp.join(self.processed_dir, 'pre_transform.pt')
#         if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
#             warnings.warn(
#                 f"The `pre_transform` argument differs from the one used in "
#                 f"the pre-processed version of this dataset. If you want to "
#                 f"make use of another pre-processing technique, make sure to "
#                 f"delete '{self.processed_dir}' first")

#         f = osp.join(self.processed_dir, 'pre_filter.pt')
#         if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
#             warnings.warn(
#                 "The `pre_filter` argument differs from the one used in "
#                 "the pre-processed version of this dataset. If you want to "
#                 "make use of another pre-fitering technique, make sure to "
#                 "delete '{self.processed_dir}' first")

#         if files_exist(self.processed_paths):  # pragma: no cover
#             return

#         if self.log and 'pytest' not in sys.modules:
#             print('Processing...', file=sys.stderr)

#         makedirs(self.processed_dir)
#         self.process()

#         path = osp.join(self.processed_dir, 'pre_transform.pt')
#         torch.save(_repr(self.pre_transform), path)
#         path = osp.join(self.processed_dir, 'pre_filter.pt')
#         torch.save(_repr(self.pre_filter), path)

#         if self.log and 'pytest' not in sys.modules:
#             print('Done!', file=sys.stderr)

#     def __len__(self) -> int:
#         r"""The number of examples in the dataset."""
#         return len(self.indices())

#     def __getitem__(
#         self,
#         idx: Union[int, np.integer, IndexType],
#     ) -> Union['Dataset', BaseData]:
#         r"""In case :obj:`idx` is of type integer, will return the data object
#         at index :obj:`idx` (and transforms it in case :obj:`transform` is
#         present).
#         In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
#         tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
#         bool, will return a subset of the dataset at the specified indices."""
#         if (isinstance(idx, (int, np.integer))
#                 or (isinstance(idx, Tensor) and idx.dim() == 0)
#                 or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

#             data = self.get(self.indices()[idx])
#             data = data if self.transform is None else self.transform(data)
#             return data

#         else:
#             return self.index_select(idx)

#     def index_select(self, idx: IndexType) -> 'Dataset':
#         r"""Creates a subset of the dataset from specified indices :obj:`idx`.
#         Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
#         list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
#         long or bool."""
#         indices = self.indices()

#         if isinstance(idx, slice):
#             start, stop, step = idx.start, idx.stop, idx.step
#             # Allow floating-point slicing, e.g., dataset[:0.9]
#             if isinstance(start, float):
#                 start = round(start * len(self))
#             if isinstance(stop, float):
#                 stop = round(stop * len(self))
#             idx = slice(start, stop, step)

#             indices = indices[idx]

#         elif isinstance(idx, Tensor) and idx.dtype == torch.long:
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
#             idx = idx.flatten().nonzero(as_tuple=False)
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, np.ndarray) and idx.dtype == bool:
#             idx = idx.flatten().nonzero()[0]
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, Sequence) and not isinstance(idx, str):
#             indices = [indices[i] for i in idx]

#         else:
#             raise IndexError(
#                 f"Only slices (':'), list, tuples, torch.tensor and "
#                 f"np.ndarray of dtype long or bool are valid indices (got "
#                 f"'{type(idx).__name__}')")

#         dataset = copy.copy(self)
#         dataset._indices = indices
#         return dataset

#     def shuffle(
#         self,
#         return_perm: bool = False,
#     ) -> Union['Dataset', Tuple['Dataset', Tensor]]:
#         r"""Randomly shuffles the examples in the dataset.

#         Args:
#             return_perm (bool, optional): If set to :obj:`True`, will also
#                 return the random permutation used to shuffle the dataset.
#                 (default: :obj:`False`)
#         """
#         perm = torch.randperm(len(self))
#         dataset = self.index_select(perm)
#         return (dataset, perm) if return_perm is True else dataset

#     def __repr__(self) -> str:
#         arg_repr = str(len(self)) if len(self) > 1 else ''
#         return f'{self.__class__.__name__}({arg_repr})'

#     def get_summary(self):
#         r"""Collects summary statistics for the dataset."""
#         from torch_geometric.data.summary import Summary
#         return Summary.from_dataset(self)

#     def print_summary(self):  # pragma: no cover
#         r"""Prints summary statistics of the dataset to the console."""
#         print(str(self.get_summary()))

#     def to_datapipe(self):
#         r"""Converts the dataset into a :class:`torch.utils.data.DataPipe`.

#         The returned instance can then be used with :pyg:`PyG's` built-in
#         :class:`DataPipes` for baching graphs as follows:

#         .. code-block:: python

#             from torch_geometric.datasets import QM9

#             dp = QM9(root='./data/QM9/').to_datapipe()
#             dp = dp.batch_graphs(batch_size=2, drop_last=True)

#             for batch in dp:
#                 pass

#         See the `PyTorch tutorial
#         <https://pytorch.org/data/main/tutorial.html>`_ for further background
#         on DataPipes.
#         """
#         from torch_geometric.data.datapipes import DatasetAdapter

#         return DatasetAdapter(self)


# def overrides_method(cls, method_name: str):
#     from torch_geometric.data import InMemoryDataset

#     if method_name in cls.__dict__:
#         return True

#     out = False
#     for base in cls.__bases__:
#         if base != Dataset and base != InMemoryDataset:
#             out |= overrides_method(base, method_name)
#     return out


# def to_list(value: Any) -> Sequence:
#     if isinstance(value, Sequence) and not isinstance(value, str):
#         return value
#     else:
#         return [value]


# def files_exist(files: List[str]) -> bool:
#     # NOTE: We return `False` in case `files` is empty, leading to a
#     # re-processing of files on every instantiation.
#     return len(files) != 0 and all([osp.exists(f) for f in files])


# def _repr(obj: Any) -> str:
#     if obj is None:
#         return 'None'
#     return re.sub('(<.*?)\\s.*(>)', r'\1\2', str(obj))


# def _get_flattened_data_list(data_list: List[Any]) -> List[BaseData]:
#     outs: List[BaseData] = []
#     for data in data_list:
#         if isinstance(data, BaseData):
#             outs.append(data)
#         elif isinstance(data, (tuple, list)):
#             outs.extend(_get_flattened_data_list(data))
#         elif isinstance(data, dict):
#             outs.extend(_get_flattened_data_list(data.values()))
#     return outs


# class InMemoryDataset(Dataset, ABC):
#     r"""Dataset base class for creating graph datasets which easily fit
#     into CPU memory.
#     See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
#     create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
#     tutorial.

#     Args:
#         root (str, optional): Root directory where the dataset should be saved.
#             (optional: :obj:`None`)
#         transform (callable, optional): A function/transform that takes in a
#             :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             transformed version.
#             The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             a :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             transformed version.
#             The data object will be transformed before being saved to disk.
#             (default: :obj:`None`)
#         pre_filter (callable, optional): A function that takes in a
#             :class:`~torch_geometric.data.Data` or
#             :class:`~torch_geometric.data.HeteroData` object and returns a
#             boolean value, indicating whether the data object should be
#             included in the final dataset. (default: :obj:`None`)
#         log (bool, optional): Whether to print any console output while
#             downloading and processing the dataset. (default: :obj:`True`)
#     """
#     @property
#     def raw_file_names(self) -> Union[str, List[str], Tuple]:
#         raise NotImplementedError

#     @property
#     def processed_file_names(self) -> Union[str, List[str], Tuple]:
#         raise NotImplementedError

#     def __init__(
#         self,
#         root: Optional[str] = None,
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#         pre_filter: Optional[Callable] = None,
#         log: bool = True,
#     ):
#         super().__init__(root, transform, pre_transform, pre_filter, log)
#         self._data = None
#         self.slices = None
#         self._data_list: Optional[List[BaseData]] = None

#     @property
#     def num_classes(self) -> int:
#         if self.transform is None:
#             return self._infer_num_classes(self._data.y)
#         return super().num_classes

#     def len(self) -> int:
#         if self.slices is None:
#             return 1
#         for _, value in nested_iter(self.slices):
#             return len(value) - 1
#         return 0

#     def get(self, idx: int) -> BaseData:
#         # TODO (matthias) Avoid unnecessary copy here.
#         if self.len() == 1:
#             return copy.copy(self._data)

#         if not hasattr(self, '_data_list') or self._data_list is None:
#             self._data_list = self.len() * [None]
#         elif self._data_list[idx] is not None:
#             return copy.copy(self._data_list[idx])

#         data = separate(
#             cls=self._data.__class__,
#             batch=self._data,
#             idx=idx,
#             slice_dict=self.slices,
#             decrement=False,
#         )

#         self._data_list[idx] = copy.copy(data)

#         return data

#     @classmethod
#     def save(cls, data_list: List[BaseData], path: str):
#         r"""Saves a list of data objects to the file path :obj:`path`."""
#         data, slices = cls.collate(data_list)
#         torch.save((data.to_dict(), slices), path)

#     def load(self, path: str, data_cls: Type[BaseData] = Data):
#         r"""Loads the dataset from the file path :obj:`path`."""
#         data, self.slices = torch.load(path)
#         if isinstance(data, dict):  # Backward compatibility.
#             data = data_cls.from_dict(data)
#         self.data = data

#     @staticmethod
#     def collate(
#         data_list: List[BaseData],
#     ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
#         r"""Collates a Python list of :class:`~torch_geometric.data.Data` or
#         :class:`~torch_geometric.data.HeteroData` objects to the internal
#         storage format of :class:`~torch_geometric.data.InMemoryDataset`."""
#         if len(data_list) == 1:
#             return data_list[0], None

#         data, slices, _ = collate(
#             data_list[0].__class__,
#             data_list=data_list,
#             increment=False,
#             add_batch=False,
#         )

#         return data, slices

#     def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
#         r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
#         will clone the full dataset. Otherwise, will only clone a subset of the
#         dataset from indices :obj:`idx`.
#         Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
#         :obj:`np.ndarray` of type long or bool.
#         """
#         if idx is None:
#             data_list = [self.get(i) for i in self.indices()]
#         else:
#             data_list = [self.get(i) for i in self.index_select(idx).indices()]

#         dataset = copy.copy(self)
#         dataset._indices = None
#         dataset._data_list = None
#         dataset.data, dataset.slices = self.collate(data_list)
#         return dataset

#     def to_on_disk_dataset(
#         self,
#         root: Optional[str] = None,
#         backend: str = 'sqlite',
#         log: bool = True,
#     ) -> 'torch_geometric.data.OnDiskDataset':
#         r"""Converts the :class:`InMemoryDataset` to a :class:`OnDiskDataset`
#         variant. Useful for distributed training and hardware instances with
#         limited amount of shared memory.

#         root (str, optional): Root directory where the dataset should be saved.
#             If set to :obj:`None`, will save the dataset in
#             :obj:`root/on_disk`.
#             Note that it is important to specify :obj:`root` to account for
#             different dataset splits. (optional: :obj:`None`)
#         backend (str): The :class:`Database` backend to use.
#             (default: :obj:`"sqlite"`)
#         log (bool, optional): Whether to print any console output while
#             processing the dataset. (default: :obj:`True`)
#         """
#         if root is None and (self.root is None or not osp.exists(self.root)):
#             raise ValueError(f"The root directory of "
#                              f"'{self.__class__.__name__}' is not specified. "
#                              f"Please pass in 'root' when creating on-disk "
#                              f"datasets from it.")

#         root = root or osp.join(self.root, 'on_disk')

#         in_memory_dataset = self
#         ref_data = in_memory_dataset.get(0)
#         if not isinstance(ref_data, Data):
#             raise NotImplementedError(
#                 f"`{self.__class__.__name__}.to_on_disk_dataset()` is "
#                 f"currently only supported on homogeneous graphs")

#         # Parse the schema ====================================================

#         schema: Dict[str, Any] = {}
#         for key, value in ref_data.to_dict().items():
#             if isinstance(value, (int, float, str)):
#                 schema[key] = value.__class__
#             elif isinstance(value, Tensor) and value.dim() == 0:
#                 schema[key] = dict(dtype=value.dtype, size=(-1, ))
#             elif isinstance(value, Tensor):
#                 size = list(value.size())
#                 size[ref_data.__cat_dim__(key, value)] = -1
#                 schema[key] = dict(dtype=value.dtype, size=tuple(size))
#             else:
#                 schema[key] = object

#         # Create the on-disk dataset ==========================================

#         class OnDiskDataset(torch_geometric.data.OnDiskDataset):
#             def __init__(
#                 self,
#                 root: str,
#                 transform: Optional[Callable] = None,
#             ):
#                 super().__init__(
#                     root=root,
#                     transform=transform,
#                     backend=backend,
#                     schema=schema,
#                 )

#             def process(self):
#                 _iter = [
#                     in_memory_dataset.get(i)
#                     for i in in_memory_dataset.indices()
#                 ]
#                 if log:  # pragma: no cover
#                     _iter = tqdm(_iter, desc='Converting to OnDiskDataset')

#                 data_list: List[Data] = []
#                 for i, data in enumerate(_iter):
#                     data_list.append(data)
#                     if i + 1 == len(in_memory_dataset) or (i + 1) % 1000 == 0:
#                         self.extend(data_list)
#                         data_list = []

#             def serialize(self, data: Data) -> Dict[str, Any]:
#                 return data.to_dict()

#             def deserialize(self, data: Dict[str, Any]) -> Data:
#                 return Data.from_dict(data)

#             def __repr__(self) -> str:
#                 arg_repr = str(len(self)) if len(self) > 1 else ''
#                 return (f'OnDisk{in_memory_dataset.__class__.__name__}('
#                         f'{arg_repr})')

#         return OnDiskDataset(root, transform=in_memory_dataset.transform)

#     @property
#     def data(self) -> Any:
#         msg1 = ("It is not recommended to directly access the internal "
#                 "storage format `data` of an 'InMemoryDataset'.")
#         msg2 = ("The given 'InMemoryDataset' only references a subset of "
#                 "examples of the full dataset, but 'data' will contain "
#                 "information of the full dataset.")
#         msg3 = ("The data of the dataset is already cached, so any "
#                 "modifications to `data` will not be reflected when accessing "
#                 "its elements. Clearing the cache now by removing all "
#                 "elements in `dataset._data_list`.")
#         msg4 = ("If you are absolutely certain what you are doing, access the "
#                 "internal storage via `InMemoryDataset._data` instead to "
#                 "suppress this warning. Alternatively, you can access stacked "
#                 "individual attributes of every graph via "
#                 "`dataset.{attr_name}`.")

#         msg = msg1
#         if self._indices is not None:
#             msg += f' {msg2}'
#         if self._data_list is not None:
#             msg += f' {msg3}'
#             self._data_list = None
#         msg += f' {msg4}'

#         warnings.warn(msg)

#         return self._data

#     @data.setter
#     def data(self, value: Any):
#         self._data = value
#         self._data_list = None

#     def __getattr__(self, key: str) -> Any:
#         data = self.__dict__.get('_data')
#         if isinstance(data, Data) and key in data:
#             if self._indices is None and data.__inc__(key, data[key]) == 0:
#                 return data[key]
#             else:
#                 data_list = [self.get(i) for i in self.indices()]
#                 return Batch.from_data_list(data_list)[key]

#         raise AttributeError(f"'{self.__class__.__name__}' object has no "
#                              f"attribute '{key}'")


# def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
#     if isinstance(node, Mapping):
#         for key, value in node.items():
#             for inner_key, inner_value in nested_iter(value):
#                 yield inner_key, inner_value
#     elif isinstance(node, Sequence):
#         for i, inner_value in enumerate(node):
#             yield i, inner_value
#     else:
#         yield None, node
        
        
class TPUGraphs(InMemoryDataset):

    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 source: str = 'nlp',  # 'nlp' or 'xla'
                 search: str = 'random'  # 'random' or 'default'
                ):
        source = cfg.source
        search = cfg.search
        print(f"Loading source {source}...")
        print(f"Loading search {search}...")
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        self.thres = thres
        self.source = source
        self.search = search
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std[op_feats_std < 1e-6] = 1
        self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        
    
    @property
    def processed_dir(self) -> str:
        tmp = osp.join(self.root, 'processed', self.source, self.search)
        os.makedirs(tmp, exist_ok=True)
        return tmp
    
    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]
    
    @property
    def raw_file_names(self) -> List[str]:
        return [f'npz/layout/{self.source}/{self.search}']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]


    def process(self):
        data_list = []
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        graphs_cnt = 0
        parts_cnt = 0
        for raw_path in self.raw_paths:
            for split_name in split_names:
                filenames = glob.glob(osp.join(os.path.join(raw_path, split_name), '*.npz'))
                if split_name == 'test':
                    self.test_filenames = filenames
                for filename in tqdm(filenames):
                    split_dict[split_name].append(graphs_cnt)
                    np_file = dict(np.load(filename))
                    if "edge_index" not in np_file:
                        print('error in', filename)
                    edge_index = torch.tensor(np_file["edge_index"].T)
                    runtime = torch.tensor(np_file["config_runtime"])
                    op = torch.tensor(np_file["node_feat"])
                    op_code = torch.tensor(np_file["node_opcode"])
                    config_feats = torch.tensor(np_file["node_config_feat"])
                    config_feats = config_feats.view(-1, config_feats.shape[-1])
                    config_idx = torch.tensor(np_file["node_config_ids"])
                    num_config = torch.tensor(np_file["node_config_feat"].shape[0])
                    num_config_idx = torch.tensor(np_file["node_config_feat"].shape[1])
                    num_nodes = torch.tensor(np_file["node_feat"].shape[0])
                    num_parts = num_nodes // self.thres + 1
                    interval = num_nodes // num_parts
                    partptr = torch.arange(0, num_nodes, interval+1)
                    if partptr[-1] != num_nodes:
                        partptr = torch.cat([partptr, torch.tensor([num_nodes])])
                    data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, config_feats=config_feats, config_idx=config_idx,
                                num_config=num_config, num_config_idx=num_config_idx, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx = parts_cnt)
                    data_list.append(data)
                    graphs_cnt += 1
                    parts_cnt += num_parts * num_config
            torch.save(self.collate(data_list), self.processed_paths[0])
            torch.save(split_dict, self.processed_paths[1])
    def get_idx_split(self):
        return torch.load(self.processed_paths[1])

if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
