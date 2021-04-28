"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op

from typing import Optional, Tuple, Sequence

# TODO: Need dynamic attr support
class _ConstantBase(Module):
    def __init__(self, size:Sequence[int], dtype: Optional[flow.dtype] = None) -> None:
        super().__init__()
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, list, tuple)
        ), "shape should be int, list or tuple format!"
        
        if isinstance(size, (int)):
            self.shape = [size]
        else:
            self.shape = size

        self.dtype = dtype
        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Build()
        )
    
    def forward(self, floating_value, integer_value, is_floating_value, dtype, shape):
        return self._op()[0]


class Ones(_ConstantBase):
    r"""
    Returns a tensor filled with the scalar value 1, 
    with the shape defined by the variable argument `size`.

    Args:
        size (int...) – a sequence of integers defining the shape of the output tensor. 
        Can be a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow as flow

        y = flow.ones(10)
        # [1 1 1 1 1 1 1 1 1 1]
        
    """
    def forward(self):
        if self.dtype == None or self.dtype == flow.int:
            self.dtype = flow.int
            floating_value = float(0)
            integer_value = int(1)
            is_floating_value = False
        else:
            self.dtype = flow.float
            floating_value = float(1.0)
            integer_value = int(0)
            is_floating_value = True
        
        return self._op(
            floating_value = floating_value, 
            integer_value = integer_value, 
            is_floating_value = is_floating_value, 
            dtype = self.dtype, 
            shape = self.shape
        )[0]

@oneflow_export("tmp.ones")
def ones_op(size, dtype=None):
    return Ones(size, dtype)()


class Zeros(_ConstantBase):
    r"""
    Returns a tensor filled with the scalar value 0, 
    with the shape defined by the variable argument `size`.

    Args:
        size (int...) – a sequence of integers defining the shape of the output tensor. 
        Can be a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        y = flow.zeros(5)

        # [0. 0. 0. 0. 0. ]

    """
    def forward(self):
        if dtype == None or dtype == flow.float:
            self.dtype = flow.float32
            floating_value = float(0.0)
            integer_value = int(0)
            is_floating_value = True
        else:
            self.dtype = flow.int
            floating_value = float(0.0)
            integer_value = int(0)
            is_floating_value = False
        
        return self._op(
            floating_value = floating_value, 
            integer_value = integer_value, 
            is_floating_value = is_floating_value, 
            dtype = self.dtype, 
            shape = self.shape
        )[0]

@oneflow_export("tmp.zeros")
def ones_op(size, dtype=None):
    return Zeros(size, dtype)()
