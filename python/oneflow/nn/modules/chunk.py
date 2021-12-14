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
from oneflow.framework.tensor import register_tensor_op
from oneflow.ops.array_ops import parse_slice_tuple_list


@register_tensor_op("chunk")
def chunk_op(input, chunks, dim: int = 0):
    """Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor. Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.

    Args:
        input (oneflow.Tensor): The tensor to split.
        chunks (int): Number of chunks to return.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List of Tensors.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
               
        >>> np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> of_out = []
        >>> of_out = flow.chunk(input, chunks=3, dim=2)
        >>> chunks = 3
        >>> of_out_shape = []
        >>> for i in range(0, chunks):
        ...     of_out_shape.append(of_out[i].numpy().shape)
        >>> of_out_shape
        [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]

        >>> np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> of_out = []
        >>> of_out = flow.chunk(input, chunks=4, dim=3)
        >>> chunks = 4
        >>> of_out_shape = []
        >>> for i in range(0, chunks):
        ...     of_out_shape.append(of_out[i].numpy().shape)
        >>> of_out_shape
        [(5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 3)]

    """
    split_size = input.shape[dim] // chunks
    if split_size * chunks != input.shape[dim]:
        split_size = [split_size] * (chunks - 1) + [
            input.shape[dim] - split_size * (chunks - 1)
        ]
    return flow._C.split(input, split_size=split_size, dim=dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)