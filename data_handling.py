"""Data handling utilities for PyTorch workflows.

This module provides:
- lightweight preprocessing wrappers around scikit-learn transformers,
- an in-memory namedtuple-style data container,
- conversion helpers between PyTorch tensors and NumPy arrays,
- HDF5 read/write helpers for wrapped datasets.

Written by: Arindam Saha, ANU, 2025.
GitHub: arindam5aha, arindam.saha@anu.edu.au
"""

import torch
from collections import namedtuple
import copy
import random
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
    
class data_transformer():
    """Wrapper for common scikit-learn feature transformers.

    Args:
        method: Transformation strategy. Supported values are
            ``'standard'``, ``'minmax'``, and ``'normalize'``.
        bounds: Optional fitting bounds. Accepts either:
            - tuple ``(lower_bounds, upper_bounds)``, or
            - dict with values ``(lower, upper)``.

    Notes:
        If ``bounds`` is supplied, the transformer is fitted during
        initialization.
    """

    def __init__(self, method:str='standard', bounds=None):
        assert method in ['standard', 'minmax','normalize']
        if method == 'standard':
            self.transformer = StandardScaler()
        if method == 'minmax':
            self.transformer = MinMaxScaler()
        if method == 'normalize':
            self.transformer = Normalizer()
        elif not isinstance(method, str):
            self.transformer = method

        self.method = method
        self.fitted = False
        if bounds is not None:
            if isinstance(bounds, dict):
                l_bounds = []
                u_bounds = []
                for _, key_data in bounds.items():
                    l_bounds.append(key_data[0])
                    u_bounds.append(key_data[1])
            else:   
                l_bounds, u_bounds = bounds
            self.transformer.fit([l_bounds, u_bounds])
            self.fitted = True

    def __call__(self, input):
        """Transform input data.

        Args:
            input: 1D/2D array-like input.

        Returns:
            Transformed values as a squeezed NumPy array.

        Notes:
            If not fitted yet, the transformer is fitted on first call.
        """
        if self.fitted:
            return self.transformer.transform(np.atleast_2d(input)).squeeze()
        else:
            self.fitted = True
            return self.transformer.fit_transform(np.atleast_2d(input)).squeeze()

    def inv(self, input):
        """Apply inverse transformation.

        Args:
            input: Transformed input values.

        Returns:
            Inverse-transformed values as a squeezed NumPy array.

        Raises:
            NotImplementedError: If the method is ``'normalize'``.
        """
        if self.method == 'normalize':
            raise NotImplementedError
        else:
            return self.transformer.inverse_transform(np.atleast_2d(input)).squeeze()

class data_wrapper():
    """Namedtuple-based in-memory data container with utility methods.

    Each stored entry is represented as a namedtuple with keys provided via
    ``data_keys``. The class supports appending, sampling, unwrap/stack,
    optional preprocessing, and HDF5 persistence.

    Args:
        name: Name of the generated namedtuple type.
        data_keys: List of field names for each sample.
        device: Torch device for tensor conversion helpers.
        preprocess: Whether to apply preprocessing in ``wrap``.
        buffer_size: Maximum size of the rolling ``buffer``.
    """

    def __init__(self, 
                 name:str, 
                 data_keys:list, 
                 device=torch.device('cpu'),
                 preprocess=False, 
                 buffer_size=10000):
        
        assert all(isinstance(key, str) for key in data_keys)
        self.data = []
        self.nt = namedtuple(name, data_keys)
        self.device = device
        self.data_keys = data_keys
        self.preprocess = preprocess
        self.preprocessors = [copy.deepcopy(data_transformer()) for _ in range(len(data_keys))]
        self.buffer = []
        self.buffer_size = buffer_size

    def make_grp(self, input_data):
        """Convert field-wise arrays into grouped per-sample tuples.

        Args:
            input_data: Tuple of aligned field arrays.

        Returns:
            Tuple where each element contains one sample across all fields.
        """
        return tuple(tuple(torch.atleast_1d(x)) for x in zip(*torch_it(input_data)))
    
    def wrap(self, input_data):
        """Add aligned batch data into storage and buffer.

        Args:
            input_data: Tuple of lists/arrays in field order. All field
                containers must have equal length.
        """
        assert isinstance(input_data, tuple)
        assert isinstance(input_data[0], (list, np.ndarray))

        if self.preprocess:
            input_data = self.transform(input_data)

        input_data = self.make_grp(input_data)
        for x in input_data:
            self.data.append(self.nt(*x))
            self.buffer.append(self.nt(*x))
    
    def append(self, input_data):
        """Append data while maintaining buffer size limit.

        Args:
            input_data: Tuple of aligned field arrays/lists.
        """
        if len(input_data[0]) + len(self.buffer) > self.buffer_size:
            reduced_size = self.buffer_size - len(input_data[0])
            self.buffer = self.sample(reduced_size)
        self.wrap(input_data)
    
    def sample(self, size=1):
        """Randomly sample from all stored data.

        Args:
            size: Number of items to sample.

        Returns:
            A list of namedtuple samples.
        """
        return random.sample(self.data, size)
    
    def unwrap(self, buffer=False):
        """Stack stored entries back into field-wise tensors.

        Args:
            buffer: If ``True``, unwrap from ``self.buffer``; otherwise from
                ``self.data``.

        Returns:
            Namedtuple with tensors stacked along the first dimension.
        """
        if buffer:
            return self.nt(*map(torch.stack, zip(*self.buffer)))
        else:
            return self.nt(*map(torch.stack, zip(*self.data)))
        
    def transform(self, input_data):
        """Apply forward preprocessing transform per field."""
        return tuple(process(data) for process, data in zip(self.preprocessors, input_data))
    
    def inv_tranform(self, input_data):
        """Apply inverse preprocessing transform per field.

        Notes:
            The method name keeps existing spelling for compatibility.
        """
        return tuple(process.inv(data) for process, data in zip(self.preprocessors, input_data))
    
    def split(self, ratio, shuffle=False):
        """Split data into train/validation subsets.

        Args:
            ratio: Train-set fraction.
            shuffle: Whether to sample before splitting.

        Returns:
            Tuple ``(train_data, validation_data)``.
        """
        if shuffle:
            data = self.sample(len(self.data))
        train_data, validation_data = data[:int(ratio * len(data))], data[int(ratio * len(data)):]
        return train_data, validation_data

    def flush(self, all=False):
        """Clear buffer or full data storage.

        Args:
            all: If ``True``, clear both ``data`` and ``buffer``. Otherwise,
                clear only ``buffer``.
        """
        if all:
            self.data = []
        self.buffer = []

    def save(self, file_name):
        """Save stored data to an HDF5 file.

        Args:
            file_name: Destination file path.
        """
        f = h5py.File(file_name, "w")
        all_data = numfy(self.unwrap())
        if self.preprocess:
            all_data = self.inv_tranform(all_data)
        for data, key in zip(all_data, self.data_keys):
            f.create_dataset(key, data=data.tolist())
        f.close()

    def read(self, file_name, store=False):
        """Read data from an HDF5 file.

        Args:
            file_name: Source file path.
            store: If ``True``, data is wrapped into current instance.

        Returns:
            Tuple of HDF5 datasets when ``store`` is ``False``; otherwise
            returns ``None``.
        """
        f = h5py.File(file_name, "r")
        read_data = tuple(f[key] for key in self.data_keys)
        f.close()
        if store:
            self.wrap(read_data)
        else:
            return read_data

def torch_it(data, device=torch.device('cpu')):
    """Convert array-like data into torch tensor(s).

    Args:
        data: A single array-like object or tuple of array-like objects.
        device: Target torch device.

    Returns:
        Tensor or tuple of tensors using ``torch.get_default_dtype()``.
    """
    dtype = torch.get_default_dtype()
    if isinstance(data, tuple):
        return tuple(torch.tensor(x, device=device, dtype=dtype) for x in data)
    else:
        return torch.tensor(data, device=device, dtype=dtype)
    
    
def torch_it_(input):
    """Convert input to tensor on CUDA when available.

    Args:
        input: Array-like input.

    Returns:
        Tensor on ``cuda:0`` if available, else CPU.
    """
    dtype = torch.get_default_dtype()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(input, dtype=dtype, device=device)
    
def numfy(data):
    """Convert tensor(s) to NumPy array(s) on CPU.

    Args:
        data: Tensor or tuple of tensors.

    Returns:
        NumPy array or tuple of NumPy arrays.
    """
    if isinstance(data, tuple):
        return tuple(x.detach().cpu().numpy() for x in data)
    else:
        return data.detach().cpu().numpy()