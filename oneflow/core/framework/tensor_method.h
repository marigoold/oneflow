/*
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
*/

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_METHOD_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_METHOD_H_

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

class Tensor;

Maybe<bool> IsContiguous(const std::shared_ptr<Tensor>& tensor);

Maybe<MirroredTensor> ShallowCopy(const std::shared_ptr<MirroredTensor>& tensor,
                                  const std::shared_ptr<const Shape>& new_shape = nullptr,
                                  const std::shared_ptr<const Stride>& new_stride = nullptr,
                                  int64_t new_storage_offset = -1,
                                  DataType new_dtype = kInvalidDataType);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_METHOD_H_
