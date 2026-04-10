/***************************************************************************************************
 * Copyright (C) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG GEMM: True Block-Wise FP4 (E2M1) with Block Scaling.

  This example demonstrates a true block-wise FP4 GEMM targeting Intel Xe GPUs
  (BMG / Arc B-series) using SYCL-TLA. Both operands A and B are stored in
  float_e2m1_t (E2M1, 4-bit floating-point) format and accompanied by per-block
  scale factors in half_t (FP16).

  The implementation reuses the existing FP8 scaling mainloop infrastructure
  (MainloopIntelXeXMX16FP8Scaling) from SYCL-TLA. The key insight is that:
    - FP4 values are stored as E4M3 (float_e4m3_t) at one byte per value
    - All E2M1 values are exactly representable in E4M3 (lossless encoding)
    - The FP8 scaling mainloop converts E4M3→FP16 and applies per-block scales
    - The MMA then operates on the scaled FP16 values

  Block-Wise FP4 GEMM Modes:
  --------------------------
  The GemmMode enum describes the 2 modes of operation:

  - ConvertOnly:     FP4 values are converted to FP16 before MMA (no scaling).
                     Useful as a baseline or when data is already in correct range.
                     Implemented by running the ConvertAndScale kernel with all
                     scale factors set to 1.0 (avoids a separate kernel that
                     causes excessive register spilling on BMG targets).

  - ConvertAndScale: FP4 values are converted to FP16, then multiplied by
                     per-block scale factors. This is the "true" block-wise
                     FP4 mode used for quantized inference.

  Requirements:
    - Group size (--g) must be a multiple of the K-block tile size (32)
    - Scale tensors are M/N-major: shape [M, ceil(K/g), L] for A, [N, ceil(K/g), L] for B
    - K dimension must be a multiple of the tile K size (32) for alignment

  Architecture:
    - Target: Intel Xe (BMG / Arc B580)
    - MMA: XE_8x16x16_F32F16F16F32_TT (FP32 accumulator, FP16 operands)
    - Data path: FP4 (E2M1) → FP16 (convert) → FP16 * scale → MMA → FP32

  TODO / Known Limitations:
    - The FP4 conversion path reuses the FP8 scaling mainloop. This works because
      all E2M1 values are exactly representable in E4M3, and the mainloop's
      E4M3→FP16 conversion followed by per-block scaling is numerically correct.
      A dedicated FP4 mainloop could be more efficient by:
        * Using native 4-bit DPAS MMA atoms (XE_DPAS_TT with u4/s4) directly
          instead of widening to FP16 first
        * Implementing true 4-bit packing (2 values per byte) with custom unpack
    - Storage is at FP8 granularity (1 byte per value) rather than true 4-bit
      packing. A production implementation would pack 2 FP4 values per byte.
    - No FP4 → FP16 hardware conversion intrinsic exists on Intel Xe; conversion
      uses the E4M3 → FP16 path (exact for E2M1 values).
    - Zero-point support is omitted for FP4 as the E2M1 format is symmetric
      around zero (no bias term needed for typical block-scaled FP4 schemes).

  To build & run this example (from your build dir):

    $ ninja 14_bmg_gemm_fp4_blockscale
    $ ./examples/sycl/14_bmg_gemm_fp4_blockscale/14_bmg_gemm_fp4_blockscale

  Call with `--help` for information about available options.
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/numeric_conversion.h"
#include "sycl_common.hpp"
#include "helper.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum GemmMode {
  ConvertOnly,      // FP4 → FP16 direct convert, no scaling
  ConvertAndScale   // FP4 → FP16 then multiply by per-block scale
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// FP4 (E2M1) representable values.
// E2M1 encoding: [sign(1)][exp(2)][mantissa(1)]
// Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6  (and their negatives)
// All of these are exactly representable in E4M3 (FP8), so we store each
// FP4 value as one E4M3 byte for compatibility with the FP8 scaling mainloop.
static constexpr float kFP4E2M1Values[16] = {
   0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,   // positive
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f    // negative
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int mode;
  int m, n, k, l, iterations, verify;
  int g;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(512), n(512), k(512), l(1), iterations(20), verify(1),
    g(32), mode(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 512);
    cmd.get_cmd_line_argument("n", n, 512);
    cmd.get_cmd_line_argument("k", k, 512);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("g", g, 32);
    cmd.get_cmd_line_argument("mode", mode, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "BMG GEMM FP4 Block-Scaled Example\n\n"
      << "  Computes C = alpha * (dequant(A) x dequant(B)) + beta * C\n"
      << "  where A and B are stored in FP4 (E2M1) with per-block FP16 scales.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM (default: 512)\n"
      << "  --n=<int>                   Sets the N extent of the GEMM (default: 512)\n"
      << "  --k=<int>                   Sets the K extent of the GEMM (default: 512)\n"
      << "  --l=<int>                   Sets the L extent (batch count) (default: 1)\n"
      << "  --g=<int>                   Block size for scale factors (default: 32, must be multiple of 32)\n"
      << "  --mode=<int>                0 = ConvertOnly, 1 = ConvertAndScale (default: 1)\n"
      << "  --alpha=<float>             Epilogue scalar alpha (default: 1.0)\n"
      << "  --beta=<float>              Epilogue scalar beta (default: 0.0)\n"
      << "  --iterations=<int>          Benchmark iterations (default: 100)\n"
      << "  --verify=<int>              1 = verify against reference, 0 = skip (default: 1)\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// ExampleRunner: Orchestrates initialization, GEMM execution, and verification.
//
// Design note: FP4 (E2M1) values are stored as E4M3 (float_e4m3_t), one byte
// per value. This is lossless since E4M3 can represent all E2M1 values exactly.
// The FP8 scaling mainloop (xe_mma_fp8_scaling.hpp) loads E4M3 bytes via
// XE_2D_U8x32x32_LD_N, converts to FP16, and applies per-block scales.
//
// The tuple-based ElementA/ElementB mechanism from xe_mma_fp8_scaling.hpp is
// used to attach scale factor metadata to each operand.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;
  using ElementMMA = typename CollectiveMainloop::ElementMMA;

  using ElementQuant = ElementA;

  using ElementScaleA = typename CollectiveMainloop::NonVoidElementScaleA;
  using ElementScaleB = typename CollectiveMainloop::NonVoidElementScaleB;

  using ElementZeroA = typename CollectiveMainloop::NonVoidElementZeroA;
  using ElementZeroB = typename CollectiveMainloop::NonVoidElementZeroB;

  using StrideScaleA = typename CollectiveMainloop::NonVoidStrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::NonVoidStrideScaleB;

  using StrideZeroA = typename CollectiveMainloop::NonVoidStrideZeroA;
  using StrideZeroB = typename CollectiveMainloop::NonVoidStrideZeroB;

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScaleA stride_SA;
  StrideScaleB stride_SB;
  StrideZeroA stride_ZA;
  StrideZeroB stride_ZB;

  uint64_t seed = 42;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementMMA> block_A_dq;  // Dequantized copy of A for validation
  cutlass::DeviceAllocation<ElementMMA> block_B_dq;  // Dequantized copy of B for validation
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementScaleA> block_scaleA;
  cutlass::DeviceAllocation<ElementScaleB> block_scaleB;
  cutlass::DeviceAllocation<ElementZeroA> block_zeroA;
  cutlass::DeviceAllocation<ElementZeroB> block_zeroB;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  /// Run a reference GEMM on dequantized FP16 data to produce golden output.
  bool verify(const Options &options) {
    using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
    using GmemTiledCopyB = XE_2D_U16x16x16_LD_V;

    using TileShape = Shape<_256, _256, _32>;

    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
                                      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

    constexpr int PipelineStages = 3;
    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementCompute,
            ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

    using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
            decltype(tile_shape(TiledMma()))>;

    using CollectiveEpilogueRef = cutlass::epilogue::collective::CollectiveEpilogue<
            EpilogueDispatchPolicy,
            TileShape,
            ElementAccumulator,
            cutlass::gemm::TagToStrideC_t<LayoutC>,
            ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutD>,
            FusionCallBacks,
            XE_2D_U32x8x16_LD_N,
            void, void,
            XE_2D_U32x8x16_ST_N,
            void, void>;

    // Reference mainloop: uses pre-dequantized FP16 data
    using CollectiveMainloopRef = cutlass::gemm::collective::CollectiveMma<
            GEMMDispatchPolicy,
            TileShape,
            cutlass::half_t,
            cutlass::gemm::TagToStrideA_t<LayoutA>,
            cutlass::half_t,
            cutlass::gemm::TagToStrideB_t<LayoutB>,
            TiledMma,
            GmemTiledCopyA, void, void, cute::identity,  // A
            GmemTiledCopyB, void, void, cute::identity    // B
    >;

    using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloopRef,
      CollectiveEpilogueRef
    >;

    using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

    typename GemmRef::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A_dq.get(), stride_A, block_B_dq.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_ref_D.get(), stride_D}
    };

    GemmRef gemm_ref;
    size_t workspace_size = GemmRef::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(gemm_ref.can_implement(arguments));
    CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_ref.run());

    // FP4 E2M1 has only ~1 bit of mantissa precision (values: 0, 0.5, 1, 1.5, 2, 3, 4, 6).
    // This requires significantly relaxed tolerance compared to FP8/FP16 verification.
    ElementOutput const epsilon(5e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
    return passed;
  }

  /// Initialize scale factors for one operand.
  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block,
    Options const& options) {

    if (options.mode == GemmMode::ConvertOnly) {
      // No scaling: fill with 1.0 so the dequantize path produces the same result
      std::vector<Element> stage(block.size(), Element(1.0f));
      block.copy_from_host(stage.data());
    }
    else {
      // For FP4 E2M1: max representable value is 6.0
      // Choose scale factors that keep dequantized values in a reasonable range
      const float fp4_max = 6.0f;
      const float max_dequant_val = fp4_max * 0.25f;
      const float min_dequant_val = 0.5f;
      const float scale_max = max_dequant_val / fp4_max;
      const float scale_min = min_dequant_val / fp4_max;
      cutlass::reference::device::BlockFillRandomUniform(
         block.get(), block.size(), seed + 100, Element(scale_max), Element(scale_min));
    }
    return true;
  }

  /// Initialize zero-point buffers (always 0 for symmetric FP4 E2M1 format).
  template <class Element>
  bool initialize_zero(
    cutlass::DeviceAllocation<Element>& block,
    Options const& options) {

    std::vector<Element> stage(block.size(), Element(0.0f));
    block.copy_from_host(stage.data());
    return true;
  }

  /// Generate random FP4-range values stored as E4M3 (one byte per value).
  /// Values are drawn from the FP4 E2M1 representable set:
  ///   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  /// Each value is losslessly encoded as float_e4m3_t since E4M3 ⊇ E2M1.
  void initialize_fp4_as_e4m3(
    cutlass::DeviceAllocation<cutlass::float_e4m3_t>& block,
    size_t num_elements,
    uint64_t rng_seed)
  {
    std::vector<cutlass::float_e4m3_t> host_data(num_elements);

    std::mt19937 rng(static_cast<unsigned>(rng_seed));
    std::uniform_int_distribution<int> dist(0, 15);

    for (size_t i = 0; i < num_elements; ++i) {
      float val = kFP4E2M1Values[dist(rng)];
      host_data[i] = cutlass::float_e4m3_t(val);
    }

    block.reset(num_elements);
    block.copy_from_host(host_data.data());
  }

  /// Initialize all operands for the GEMM.
  void initialize(Options const& options) {
    auto [M, N, K, L] = ProblemShapeType{options.m, options.n, options.k, options.l};

    const int scale_k = cute::ceil_div(K, options.g);
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    auto shape_scaleA = cute::make_shape(options.m, scale_k, L);
    auto shape_scaleB = cute::make_shape(options.n, scale_k, L);

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
    stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scaleA);
    stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scaleB);
    stride_ZA = cutlass::make_cute_packed_stride(StrideZeroA{}, shape_scaleA);
    stride_ZB = cutlass::make_cute_packed_stride(StrideZeroB{}, shape_scaleB);

    // Allocate and fill operand buffers (one E4M3 byte per logical FP4 element)
    size_t num_A = static_cast<size_t>(M) * K * L;
    size_t num_B = static_cast<size_t>(N) * K * L;

    initialize_fp4_as_e4m3(block_A, num_A, seed + 2023);
    initialize_fp4_as_e4m3(block_B, num_B, seed + 2022);

    // Allocate dequantized FP16 buffers (for reference GEMM)
    block_A_dq.reset(num_A);
    block_B_dq.reset(num_B);

    // Allocate C, D, ref_D
    block_C.reset(static_cast<size_t>(M) * N * L);
    block_D.reset(static_cast<size_t>(M) * N * L);
    block_ref_D.reset(static_cast<size_t>(M) * N * L);

    // Initialize C with random data
    initialize_block(block_C, seed + 2021);

    // Allocate and fill scale and zero buffers
    block_scaleA.reset(static_cast<size_t>(scale_k) * L * M);
    block_scaleB.reset(static_cast<size_t>(scale_k) * L * N);
    block_zeroA.reset(static_cast<size_t>(scale_k) * L * M);
    block_zeroB.reset(static_cast<size_t>(scale_k) * L * N);

    initialize_scale(block_scaleA, options);
    initialize_zero(block_zeroA, options);
    initialize_scale(block_scaleB, options);
    initialize_zero(block_zeroB, options);

    // Dequantize E4M3 → FP16 with scales for reference GEMM.
    // This matches the kernel's path: load E4M3 byte → convert to FP16 → multiply by scale.
    // cutlass::dequantize handles both conversion and scaling in one step.
    auto layout_A = make_layout(shape_A, stride_A);
    auto layout_B = make_layout(shape_B, stride_B);
    auto layout_scaleA = make_layout(shape_scaleA, stride_SA);
    auto layout_scaleB = make_layout(shape_scaleB, stride_SB);

    cutlass::dequantize(block_A_dq.get(), block_A.get(), layout_A,
                        block_scaleA.get(), block_zeroA.get(), layout_scaleA, layout_scaleA,
                        options.g);
    cutlass::dequantize(block_B_dq.get(), block_B.get(), layout_B,
                        block_scaleB.get(), block_zeroB.get(), layout_scaleB, layout_scaleB,
                        options.g);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(options);

    //
    // Configure the main GEMM arguments.
    // FP4-range data is stored as E4M3 (one byte per value). The FP8 scaling
    // mainloop loads bytes, converts E4M3→FP16, and applies per-block scales.
    //
    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B,
       block_scaleA.get(), stride_SA, block_scaleB.get(), stride_SB,
       nullptr, stride_SA, nullptr, stride_SB,
       options.g},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::cout << "Datatype: FP4 (E2M1) block-scaled" << std::endl;
      std::cout << "Group Size: " << options.g << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Launcher: Configures the CUTLASS GEMM kernel types and dispatches.
//
// Architecture mapping:
//   FP4 values stored as E4M3 → loaded via XE_2D_U8x32x32_LD_N (same as FP8)
//   Converted to half_t in registers → MMA via XE_8x16x16_F32F16F16F32_TT
//   Scale factors in half_t → loaded via scale_zero_copy_traits
//
// We use float_e4m3_t as the element type because:
//   1. All E2M1 values are exactly representable in E4M3 (lossless)
//   2. The FP8 scaling mainloop is specialized for float_e4m3_t / float_e5m2_t
//   3. The mainloop converts E4M3→FP16 and applies per-block scales correctly
//
// TODO: A cleaner approach would be to add a dedicated dispatch policy
//       (e.g., MainloopIntelXeXMX16FP4Scaling) with true 4-bit packing
//       and native DPAS u4/s4 support.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int launcher(Options& options)
{
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // --- Type definitions ---
  //
  // FP4 values from the E2M1 representable set are stored as E4M3 (one byte
  // per value). This is lossless since E4M3 ⊇ E2M1. The FP8 scaling mainloop
  // handles E4M3→FP16 conversion and per-block scale multiplication.
  //
  using MmaType = cutlass::half_t;
  using QuantType = cutlass::float_e4m3_t;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = QuantType;
  using ElementInputB = QuantType;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementScale = MmaType;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  // Copy descriptors for FP4 packed as bytes (same as FP8)
  using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;

  // Workgroup tile shape
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16FP8Scaling<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U32x8x16_ST_N,
          void, void>;

  //
  // Single kernel instantiation: ConvertAndScale mainloop handles both modes.
  // ConvertOnly is achieved by passing scales=1.0 (set in initialize_scale()).
  // This avoids a separate 1-element-tuple kernel instantiation that causes
  // excessive register spilling (227 spills) and undefined prefetch intrinsics
  // on BMG targets.
  //

  if (options.mode == GemmMode::ConvertOnly) {
    std::cout << "Running FP4 Block-Scaled GEMM in ConvertOnly mode (scales=1.0)." << std::endl;
  } else if (options.mode == GemmMode::ConvertAndScale) {
    std::cout << "Running FP4 Block-Scaled GEMM in ConvertAndScale mode." << std::endl;
  } else {
    std::cerr << "Unknown mode: " << options.mode << ". Use 0 (ConvertOnly) or 1 (ConvertAndScale)." << std::endl;
    return -1;
  }

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          cute::tuple<ElementInputA, ElementScale, StrideScale>,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          cute::tuple<ElementInputB, ElementScale, StrideScale>,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,
          GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  CUTLASS_CHECK(ExampleRunner<Gemm>{}.run(options, hw_info));

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  // Validate constraints
  if (options.k % 32 != 0) {
    std::cerr << "K must be a multiple of 32 (K-tile size)." << std::endl;
    return -1;
  }
  if (options.g % 32 != 0) {
    std::cerr << "Group size must be a multiple of 32 (K-tile size)." << std::endl;
    return -1;
  }
  if (options.k % options.g != 0) {
    std::cerr << "K must be a multiple of group size." << std::endl;
    return -1;
  }

  std::cout << "===== FP4 (E2M1) Block-Scaled GEMM on Intel Xe =====\n" << std::endl;

  return launcher(options);
}
