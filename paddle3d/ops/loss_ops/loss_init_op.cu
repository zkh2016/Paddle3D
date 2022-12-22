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

#include "paddle/extension.h"

__global__ void init_idx( 
    const int n,
    const int len,
    int64_t* inputs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = blockIdx.y;
    if(bid < n && tid < len){
        int64_t* ptr = (int64_t*)inputs[bid];
        ptr[tid] = bid;
    }
}

std::vector<paddle::Tensor> loss_init_cuda(const paddle::Tensor& in){
    const int m = in.shape()[0];
    const int n = in.shape()[1];
   std::vector<paddle::Tensor> inputs(n);
   std::vector<int64_t*> input_ptrs(n);
   for(int i = 0; i < n; ++i){
    inputs[i] = paddle::empty({m, 1, 1}, paddle::DataType::INT64, paddle::GPUPlace());
    input_ptrs[i] = inputs[i].data<int64_t>();
   }
   paddle::Tensor d_ptrs = paddle::empty({n},
           paddle::DataType::INT64, paddle::GPUPlace());
   cudaMemcpyAsync(d_ptrs.data<int64_t>(), input_ptrs.data(),
           sizeof(int64_t*) * n, cudaMemcpyHostToDevice,
           inputs[0].stream()); 

   dim3 blocks(256, 1, 1);
   dim3 grids((m + 255) / 256, n, 1);  
   init_idx<<<grids, blocks, 0, inputs[0].stream()>>>(n, m,
           d_ptrs.data<int64_t>());
   return {paddle::experimental::concat(inputs, 1)}; 
}


