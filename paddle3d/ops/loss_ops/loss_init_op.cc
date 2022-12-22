// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> loss_init_cpu(const paddle::Tensor& in){
    const int m = in.shape()[0];
    const int n = in.shape()[1];
   paddle::Tensor ret = paddle::empty({n, m, 1, 1},
           paddle::DataType::INT64, paddle::CPUPlace()); 
   return {ret};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> loss_init_cuda(const paddle::Tensor& in);
#endif

std::vector<paddle::Tensor> loss_init(const paddle::Tensor& in){
#ifdef PADDLE_WITH_CUDA
    return loss_init_cuda(in);
#else
    return loss_init_cpu(in);
#endif
}

std::vector<std::vector<int64_t>> LossInferShape(
    const std::vector<int64_t> in_shape) {
    return {{in_shape[1], in_shape[0], 1, 1}};
}
std::vector<paddle::DataType>
LossInferDtype(paddle::DataType in_dtype) {
    return {paddle::DataType::INT64};
}

PD_BUILD_OP(loss_init)
    .Inputs({"X"})
    .Outputs({"Y"})
    .SetKernelFn(PD_KERNEL(loss_init))
    .SetInferShapeFn(PD_INFER_SHAPE(LossInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LossInferDtype));
