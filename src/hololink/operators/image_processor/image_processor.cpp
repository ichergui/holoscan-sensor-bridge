/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "image_processor.hpp"
#include "image_processor_kernels.cuh"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <holoscan/holoscan.hpp>

namespace {

// 3 channels (RGB)
constexpr auto CHANNELS = 3;
// histogram bin's
constexpr auto HISTOGRAM_BIN_COUNT = 256;

} // anonymous namespace

namespace hololink::operators {

ImageProcessorOp::~ImageProcessorOp() = default;

void ImageProcessorOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(bayer_format_, "bayer_format", "BayerFormat", "Bayer format (one of hololink::csi::BayerFormat)");
    spec.param(pixel_format_, "pixel_format", "PixelFormat", "Pixel format (one of hololink::csi::PixelFormat)");
    spec.param(optical_black_, "optical_black", "Optical Black", "optical black value", 0);
    spec.param(
        cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "Device to use for CUDA operations", 0);
    cuda_stream_handler_.define_params(spec);
}

void ImageProcessorOp::start()
{
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    // histogram setup
    const auto log2_warp_size = 5;
    const auto warp_size = 1 << log2_warp_size;

    // size of histogram memory
    constexpr auto histogram_warp_memory = HISTOGRAM_BIN_COUNT * sizeof(uint32_t) * CHANNELS;
    histogram_memory_.reset([] {
        CUdeviceptr mem = 0;
        CudaCheck(cuMemAlloc(&mem, histogram_warp_memory));
        return mem;
    }());

    // calculate the maximum warp count supported by the available shared memory
    // size (warps == subhistograms per threadblock)
    CUdevice cuda_device = 0;
    CudaCheck(cuDeviceGet(&cuda_device, cuda_device_ordinal_));
    int shm_size = 0;
    CudaCheck(cuDeviceGetAttribute(&shm_size, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuda_device));

    // round down since we can't exceed the available size
    const auto histogram_warp_count = shm_size / histogram_warp_memory;
    // the shared memory per threadblock is the memory needed by one warp
    // multiplied by the warps we launch
    const auto histogram_threadblock_memory = histogram_warp_memory * histogram_warp_count;
    // threadblock size
    histogram_threadblock_size_ = histogram_warp_count * warp_size;

    uint32_t least_significant_bit;
    switch (hololink::csi::PixelFormat(pixel_format_.get())) {
    case hololink::csi::PixelFormat::RAW_8:
        least_significant_bit = 0;
        break;
    case hololink::csi::PixelFormat::RAW_10:
        // data is stored in the upper 10 bits of a 16 bit value
        least_significant_bit = 16 - 10;
        break;
    case hololink::csi::PixelFormat::RAW_12:
        // data is stored in the upper 12 bits of a 16 bit value
        least_significant_bit = 16 - 12;
        break;
    default:
        throw std::runtime_error(fmt::format("Camera pixel format {} not supported.", int(pixel_format_.get())));
    }

    uint32_t x0y0_offset, x1y0_offset, x0y1_offset, x1y1_offset;
    switch (hololink::csi::BayerFormat(bayer_format_.get())) {
    case hololink::csi::BayerFormat::RGGB:
        x0y0_offset = 0; // R
        x1y0_offset = 1; // G
        x0y1_offset = 1; // G
        x1y1_offset = 2; // B
        break;
    case hololink::csi::BayerFormat::GBRG:
        x0y0_offset = 1; // G
        x1y0_offset = 2; // B
        x0y1_offset = 0; // R
        x1y1_offset = 1; // G
        break;
    default:
        throw std::runtime_error(fmt::format("Camera bayer format {} not supported.", int(bayer_format_.get())));
    }

    // Initialize kernel configuration
    kernel_config_.reset(new hololink::operators::kernels::KernelConfig{
        x0y0_offset,
        x1y0_offset,
        x0y1_offset,
        x1y1_offset,
        static_cast<unsigned int>(optical_black_.get() * (1 << least_significant_bit)),
        HISTOGRAM_BIN_COUNT,
        histogram_threadblock_size_,
        static_cast<unsigned int>(histogram_threadblock_memory),
        log2_warp_size,
        static_cast<unsigned int>(histogram_warp_count),
        CHANNELS
    });

    white_balance_gains_memory_.reset([] {
        CUdeviceptr mem = 0;
        CudaCheck(cuMemAlloc(&mem, CHANNELS * sizeof(float)));
        return mem;
    }());
}

void ImageProcessorOp::stop()
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    kernel_config_.reset();
    histogram_memory_.reset();
    white_balance_gains_memory_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void ImageProcessorOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output, holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format("Failed to get the CUDA stream from incoming messages: {}", GxfResultStr(stream_handler_result)));
    }

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HSB_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    if (input_tensor->rank() != 3) {
        throw std::runtime_error("Tensor must be an image");
    }
    if (input_tensor->element_type() != nvidia::gxf::PrimitiveType::kUnsigned16) {
        throw std::runtime_error(fmt::format("Unexpected image data type '{}', expected '{}'", int(input_tensor->element_type()), int(nvidia::gxf::PrimitiveType::kUnsigned16)));
    }

    const uint32_t height = input_tensor->shape().dimension(0);
    const uint32_t width = input_tensor->shape().dimension(1);
    const uint32_t components = input_tensor->shape().dimension(2);
    if (components != 1) {
        throw std::runtime_error(fmt::format("Unexpected component count {}, expected '1'", components));
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    // apply optical black if set
    if (optical_black_ != 0.f) {
        hololink::operators::kernels::launchApplyBlackLevel(
            reinterpret_cast<unsigned short*>(input_tensor->pointer()),
            width, height,
            kernel_config_->optical_black,
            *kernel_config_,
            cuda_stream);
    }

    // apply Grey World White Balance algorithm
    CudaCheck(cuMemsetD32Async(histogram_memory_.get(), 0, CHANNELS * HISTOGRAM_BIN_COUNT, cuda_stream));
    hololink::operators::kernels::launchHistogram(
        reinterpret_cast<const unsigned short*>(input_tensor->pointer()),
        reinterpret_cast<unsigned int*>(static_cast<CUdeviceptr>(histogram_memory_.get())),
        width, height,
        *kernel_config_,
        cuda_stream);

    // calculate white balance gains
    hololink::operators::kernels::launchCalcWBGains(
        reinterpret_cast<unsigned int*>(static_cast<CUdeviceptr>(histogram_memory_.get())),
        reinterpret_cast<float*>(static_cast<CUdeviceptr>(white_balance_gains_memory_.get())),
        *kernel_config_,
        cuda_stream);

    hololink::operators::kernels::launchApplyOperations(
        reinterpret_cast<unsigned short*>(input_tensor->pointer()),
        width, height,
        reinterpret_cast<const float*>(static_cast<CUdeviceptr>(white_balance_gains_memory_.get())),
        *kernel_config_,
        cuda_stream);

    // pass the CUDA stream to the output message
    auto out_message = nvidia::gxf::Expected<nvidia::gxf::Entity>(entity);
    stream_handler_result
        = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }

    // Emit the tensor
    output.emit(entity);
}

} // namespace hololink::operators
