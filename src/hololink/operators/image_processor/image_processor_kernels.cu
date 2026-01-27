/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "image_processor_kernels.cuh"
#include <device_atomic_functions.h>
#include <cooperative_groups.h>

namespace hololink::operators::kernels {

// Device-side constant memory for kernel configuration
__constant__ KernelConfig d_config;

// bayer component offsets
__inline__ __device__ unsigned int getBayerOffset(unsigned int x, unsigned int y)
{
    const unsigned int offsets[2][2]{{d_config.x0y0_offset, d_config.x1y0_offset}, 
                                     {d_config.x0y1_offset, d_config.x1y1_offset}};
    return offsets[y & 1][x & 1];
}

/**
 * Apply black level correction.
 */
__global__ void applyBlackLevel(unsigned short *image,
                                int components_per_line,
                                int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= components_per_line) || (idx_y >= height))
        return;

    const int index = idx_y * components_per_line + idx_x;

    // subtract optical black and clamp
    float value = max(float(image[index]) - float(d_config.optical_black), 0.f);
    // fix white level
    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;
    value *= range / (range - float(d_config.optical_black));
    image[index] = (unsigned short)(value + 0.5f);
}

/**
 * Calculate the histogram of an image.
 *
 * Based on the Cuda SDK histogram256 sample.
 */
__global__ void histogram(const unsigned short *in,
                          unsigned int *histogram,
                          unsigned int width,
                          unsigned int height)
{
    uint2 index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (index.y >= height)
    {
        return;
    }

    // per-warp subhistogram storage
    extern __shared__ unsigned int s_hist[];

    // clear shared memory storage for current threadblock before processing
    if (threadIdx.y == 0)
    {
        const unsigned int items_per_thread = (d_config.histogram_threadblock_memory / sizeof(unsigned int)) / d_config.histogram_threadblock_size;
        for (unsigned int i = 0; i < items_per_thread; ++i)
        {
            s_hist[threadIdx.x + i * d_config.histogram_threadblock_size] = 0;
        }
    }

    // handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    cooperative_groups::sync(cta);

    // cycle through the entire data set, update subhistograms for each warp
    unsigned int *const s_warp_hist = s_hist + (threadIdx.x >> d_config.log2_warp_size) * d_config.histogram_bin_count * d_config.channels;
    while (index.x < width)
    {
        // take the upper 8 bits
        const unsigned char bin = ((unsigned char*)&in[index.y * width + index.x])[1];
        atomicAdd(s_warp_hist + bin + getBayerOffset(index.x, index.y) * d_config.histogram_bin_count, 1u);
        index.x += blockDim.x * gridDim.x;
    }

    // Merge per-warp histograms into per-block and write to global memory
    cooperative_groups::sync(cta);

    if (threadIdx.y == 0)
    {
        for (unsigned int bin = threadIdx.x; bin < d_config.histogram_bin_count * d_config.channels; bin += d_config.histogram_threadblock_size)
        {
            unsigned int sum = 0;

            for (unsigned int i = 0; i < d_config.histogram_warp_count; ++i)
            {
                sum += s_hist[bin + i * d_config.histogram_bin_count * d_config.channels];
            }

            atomicAdd(&histogram[bin], sum);
        }
    }
}

/**
 * Calculate the white balance gains using the per channel histograms
 */
__global__ void calcWBGains(const unsigned int *histogram,
                            float *gains)
{
    unsigned long long int average[3]; // max 3 channels (RGB)
    unsigned long long int max_gain = 0;
    for (unsigned int channel = 0; channel < d_config.channels; ++channel)
    {
        unsigned long long int value = 0;
        for (unsigned int bin = 1; bin < d_config.histogram_bin_count; ++bin)
        {
            value += histogram[channel * d_config.histogram_bin_count + bin] * bin;
        }
        if (channel == 1)
        {
            // there are two green channels in the image which both are counted
            // in one histogram therefore divide green channel by 2
            value /= 2;
        }
        max_gain = max(max_gain, value);
        average[channel] = max(value, 1ull);
    }

    for (unsigned int channel = 0; channel < d_config.channels; ++channel)
    {
        gains[channel] = float(max_gain) / float(average[channel]);
    }
}

/**
 * Apply white balance gains.
 */
__global__ void applyOperations(unsigned short *image,
                               int width,
                               int height,
                               const float *gains)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int index = idx_y * width + idx_x;

    float value = (float)(image[index]);

    // apply gain
    const unsigned int channel = getBayerOffset(idx_x, idx_y);
    value *= gains[channel];

    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;

    // clamp
    value = max(min(value, range), 0.f);

    image[index] = (unsigned short)(value + 0.5f);
}

// Launcher functions

void launchApplyBlackLevel(unsigned short *image,
                           int components_per_line,
                           int height,
                           unsigned int optical_black,
                           const KernelConfig& config,
                           cudaStream_t stream)
{
    // Update constant memory with config
    cudaMemcpyToSymbolAsync(d_config, &config, sizeof(KernelConfig), 0, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((components_per_line + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    applyBlackLevel<<<grid, block, 0, stream>>>(image, components_per_line, height);
}

void launchHistogram(const unsigned short *in,
                    unsigned int *histogram_out,
                    unsigned int width,
                    unsigned int height,
                    const KernelConfig& config,
                    cudaStream_t stream)
{
    // Update constant memory with config
    cudaMemcpyToSymbolAsync(d_config, &config, sizeof(KernelConfig), 0, cudaMemcpyHostToDevice, stream);

    dim3 block(config.histogram_threadblock_size, 2, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    histogram<<<grid, block, config.histogram_threadblock_memory, stream>>>(in, histogram_out, width, height);
}

void launchCalcWBGains(const unsigned int *histogram,
                      float *gains,
                      const KernelConfig& config,
                      cudaStream_t stream)
{
    // Update constant memory with config
    cudaMemcpyToSymbolAsync(d_config, &config, sizeof(KernelConfig), 0, cudaMemcpyHostToDevice, stream);

    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    
    calcWBGains<<<grid, block, 0, stream>>>(histogram, gains);
}

void launchApplyOperations(unsigned short *image,
                          int width,
                          int height,
                          const float *gains,
                          const KernelConfig& config,
                          cudaStream_t stream)
{
    // Update constant memory with config
    cudaMemcpyToSymbolAsync(d_config, &config, sizeof(KernelConfig), 0, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    applyOperations<<<grid, block, 0, stream>>>(image, width, height, gains);
}

} // namespace hololink::operators::kernels
