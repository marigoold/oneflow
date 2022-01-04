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
#include "oneflow/user/kernels/embedding_kernel_util.h"
#include <cstdint>

namespace oneflow {

namespace {

template<typename T, typename index_T>
__global__ void embedding_kernel(const T* weight_buf, const index_T* indices_buf, T* out_buf, 
                                 const int32_t num_indices,const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, num_indices * emb_dim){
     int32_t indices_index = i / emb_dim;
     int32_t dim_index = i - i * indices_index;
     int32_t from_index = indices_buf[indices_index] + dim_index;
     out_buf[i] = weight_buf[from_index];
  }
}

template<typename T, typename index_T>
__global__ void embedding_grad_kernel(const T* dy_buf, const index_T* indices_buf, T* dx_buf, const int32_t padding_idx,
                                      const int32_t num_indices, const int32_t emb_dim) {
       CUDA_1D_KERNEL_LOOP(i, num_indices * emb_dim){
           int32_t indice = indices_buf[i];
           int32_t indices_index = i / emb_dim;
           int32_t dim_index = i - i * indices_index;
           int32_t from_index = indices_buf[indices_index] + dim_index;
           if(indice != padding_idx){
               dx_buf[from_index] += dy_buf[i];
           }
       }
  }
}

template<typename index_T>
__global__ void indices_freq(const index_T* indices_buf, const int32_t num_indices, index_T * tmp_buf){
        CUDA_1D_KERNEL_LOOP(i, num_indices){
           tmp_buf[indices_buf[i]]++;
        }
}

template<typename T, typename index_T>
__global__ void embedding_scale(T* dx_buf, const int32_t emb_size, const int32_t emb_dim, index_T * tmp_buf){
        CUDA_1D_KERNEL_LOOP(i, emb_size*emb_dim){
            int32_t emb_size_index = i/emb_dim;
            if(tmp_buf[i]>1){
                dx_buf[i]/=tmp_buf[emb_size_index];
            }
        }
}

}



template<typename T, typename index_T>
struct EmbeddingFunctor<DeviceType::kCUDA, T, index_T> final{
    void operator()(ep::Stream* stream, const T* weight_buf, const index_T* indices_buf, T* out_buf,
                    const int32_t padding_idx, const bool scale_grad_by_freq,  
                    const int32_t num_indices, const int32_t emb_dim, const int32_t emb_size){
        embedding_kernel<T, index_T>
          <<<BlocksNum4ThreadsNum(num_indices * emb_dim), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(weight_buf, indices_buf, out_buf, num_indices, emb_dim);
    }
};


template<typename T, typename index_T>
struct EmbeddingGradFunctor<DeviceType::kCUDA, T, index_T> final{
    void operator()(ep::Stream* stream, const T* dy_buf, const index_T* indices_buf, T* dx_buf,
                    const int32_t padding_idx, const bool scale_grad_by_freq,  const int32_t num_indices, const int32_t emb_dim, const int32_t emb_size
                    index_T * tmp_buf){

        embedding_grad_kernel<T, index_T>
          <<<BlocksNum4ThreadsNum(num_indices*emb_dim), kCudaThreadsNumPerBlock, 0, 
          stream->As<ep::CudaStream>()->cuda_stream()>>>(dy_buf, indices_buf, dx_buf, padding_idx, num_indices, emb_dim);
        
        if(scale_grad_by_freq){
           indices_freq<index_T><<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(indices_buf, num_indices, tmp_buf);

           embedding_scale<T, index_T><<<BlocksNum4ThreadsNum(emb_size * emb_dim), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(dx_buf, emb_size, emb_dim, tmp_buf);
        }

        ////
        for(int i=0;i<num_indices*emb_dim;i++){
           int32_t indice = indices_buf[i];
           int32_t indices_index = i / emb_dim;
           int32_t dim_index = i - i * indices_index;
           int32_t from_index = indices_buf[indices_index] + dim_index;
           if(indice != padding_idx){
               dx[from_index] += dy_buf[i]
           }
        }
        
        for(int32_t i = 0; i < num_indices; i++){
            tmp_buf[indices_buf[i]]++;
        }

        for(int i=0;i<emb_size*emb_dim;i++){
            int32_t emb_size_index = i/emb_dim;
            if(tmp_buf[i]>1){
                dx_buf[i]/=tmp_buf[emb_size_index];
            }
        }
        ////
        
        for(int32_t i = 0;i < num_indices; i++){
            int32_t indice = indices_buf[i];
            if(indice != padding_idx){
                 const T* from = dy_buf + i * emb_dim;
                 T* to = dx_buf + indice * emb_dim;
                 std::transform(from, from + emb_dim, to, to, std::plus<T>());
            }
        }

        if(scale_grad_by_freq){
            std::vector<index_T> indice_freq(emb_size, 0);
            for(int32_t i = 0; i < num_indices; i++){
                indice_freq[indices_buf[i]]++;
            }

            for(int32_t i = 0; i< emb_size;i++){
                if(indice_freq[i]>1){
                    T* from = dx_buf + i * emb_dim;
                    for(int32_t j=0; j<emb_dim; j++){
                        from[j]/=indice_freq[i];
                    }
                }
            }
        }
        
    }
};

#define INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL(in_type_pair, index_type_pair)                   \
      template struct EmbeddingRenormFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair),   \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;                       \
      template struct EmbeddingFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair),         \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;                       \
      template struct EmbeddingGradFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair),     \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL, EMBEDDING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL

}  // namespace oneflow