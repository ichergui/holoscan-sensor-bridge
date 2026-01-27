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

#ifndef SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR_KERNELS_CUH
#define SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR_KERNELS_CUH

#include <cuda_runtime.h>

namespace hololink::operators::kernels {

// Kernel configuration structure
struct KernelConfig {
    unsigned int x0y0_offset;
    unsigned int x1y0_offset;
    unsigned int x0y1_offset;
    unsigned int x1y1_offset;
    unsigned int optical_black;
    unsigned int histogram_bin_count;
    unsigned int histogram_threadblock_size;
    unsigned int histogram_threadblock_memory;
    unsigned int log2_warp_size;
    unsigned int histogram_warp_count;
    unsigned int channels;
};

/**
 * Apply black level correction.
 *
 * @param image [in] pointer to input image
 * @param components_per_line [in] components per input image line (width * 3 for RGB)
 * @param height [in] height of the input image
 * @param optical_black [in] optical black value
 */
void launchApplyBlackLevel(unsigned short *image,
                           int components_per_line,
                           int height,
                           unsigned int optical_black,
                           const KernelConfig& config,
                           cudaStream_t stream);

/**
 * Calculate the histogram of an image.
 *
 * @param in [in] pointer to image data
 * @param histogram_out [out] pointer to the histogram data
 * @param width [in] width of the image
 * @param height [in] height of the image
 * @param config [in] kernel configuration
 * @param stream [in] CUDA stream
 */
void launchHistogram(const unsigned short *in,
                    unsigned int *histogram_out,
                    unsigned int width,
                    unsigned int height,
                    const KernelConfig& config,
                    cudaStream_t stream);

/**
 * Calculate the white balance gains using the per channel histograms
 *
 * @param histogram [in] pointer to histogram data
 * @param gains [out] pointer to the white balance gains
 * @param config [in] kernel configuration
 * @param stream [in] CUDA stream
 */
void launchCalcWBGains(const unsigned int *histogram,
                      float *gains,
                      const KernelConfig& config,
                      cudaStream_t stream);

/**
 * Apply white balance gains.
 *
 * @param image [in/out] pointer to image
 * @param width [in] width of the image
 * @param height [in] height of the image
 * @param gains [in] pointer to the white balance gains
 * @param config [in] kernel configuration
 * @param stream [in] CUDA stream
 */
void launchApplyOperations(unsigned short *image,
                          int width,
                          int height,
                          const float *gains,
                          const KernelConfig& config,
                          cudaStream_t stream);

} // namespace hololink::operators::kernels

#endif /* SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR_KERNELS_CUH */
