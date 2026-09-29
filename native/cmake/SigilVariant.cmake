# SPDX-License-Identifier: AGPL-3.0-only
#
# SigilVariant.cmake - the build variant of a native engine tree (one tree per variant).
#
#   -DSIGIL_VARIANT=cpu | cpu-avx2 | cpu-avx512 | vulkan | cuda13
#
# Applies, for every target in the tree (engines and the third-party code they link statically):
# - the MSVC runtime /MD (MultiThreadedDLL; MultiThreadedDebugDLL in Debug);
# - the variant's instruction sets: cpu = SSE4.2 baseline (Windows 11 requires it), cpu-avx2 =
#   /arch:AVX2 (+ FMA, F16C, BMI2), cpu-avx512 = /arch:AVX512 (F, CD, BW, DQ, VL); the GPU variants
#   use the cpu-avx2 baseline for their CPU kernels, so Sigil only offers them on AVX2 CPUs;
# - ggml's options for in-process DLLs: static ggml inside the engine DLL (BUILD_SHARED_LIBS OFF),
#   GGML_NATIVE OFF (never the build machine's CPU), GGML_BACKEND_DL OFF (no ggml-*.dll backends
#   loaded by name), GGML_CPU_ALL_VARIANTS OFF, and the instruction-set switches of the variant
#   (FORCEd: the variant is authoritative);
# - vulkan: GGML_VULKAN ON (needs the Vulkan SDK); cuda13: GGML_CUDA ON with the CUDA 13 toolkit and
#   SIGIL_CUDA_ARCHITECTURES (Turing and newer; CUDA 13 dropped Maxwell / Pascal / Volta).
# SIGIL_VARIANT_ARCH_FLAG is the compiler flag of the variant (for build_info strings).

set(SIGIL_VARIANTS cpu cpu-avx2 cpu-avx512 vulkan cuda13)
set(SIGIL_VARIANT "cpu" CACHE STRING "Runtime variant: ${SIGIL_VARIANTS}")
set_property(CACHE SIGIL_VARIANT PROPERTY STRINGS ${SIGIL_VARIANTS})
if(NOT SIGIL_VARIANT IN_LIST SIGIL_VARIANTS)
  message(FATAL_ERROR "SIGIL_VARIANT='${SIGIL_VARIANT}' is not one of: ${SIGIL_VARIANTS}")
endif()

option(SIGIL_REPRODUCIBLE "Deterministic compiler / linker output (/Brepro, no absolute PDB path)" ON)

# --- Runtime library and language ---------------------------------------------------------------
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Third-party libraries are linked statically into each engine DLL" FORCE)

if(MSVC)
  # C/C++ only: nvcc (cuda13 trees) and rc.exe must never receive raw MSVC compiler flags.
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:C,CXX>:/Zi>")
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  if(SIGIL_REPRODUCIBLE)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/Brepro>")
    add_link_options(/Brepro)
  endif()
endif()

# --- Instruction sets -----------------------------------------------------------------------------
set(SIGIL_VARIANT_ARCH_FLAG "")
set(_sigil_avx OFF)
set(_sigil_avx2 OFF)
set(_sigil_avx512 OFF)
if(SIGIL_VARIANT STREQUAL "cpu-avx512")
  set(_sigil_avx ON)
  set(_sigil_avx2 ON)
  set(_sigil_avx512 ON)
  set(SIGIL_VARIANT_ARCH_FLAG "/arch:AVX512")
elseif(SIGIL_VARIANT STREQUAL "cpu")
  set(SIGIL_VARIANT_ARCH_FLAG "SSE4.2")
else()
  # cpu-avx2, and the CPU kernels of the GPU variants.
  set(_sigil_avx ON)
  set(_sigil_avx2 ON)
  set(SIGIL_VARIANT_ARCH_FLAG "/arch:AVX2")
endif()
if(MSVC AND SIGIL_VARIANT_ARCH_FLAG MATCHES "^/arch:")
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${SIGIL_VARIANT_ARCH_FLAG}>")
endif()

# ggml (and llama.cpp-family) options, applied when an engine adds ggml as a subdirectory.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC OFF CACHE BOOL "" FORCE)  # GGML_STATIC would switch MSVC to /MT
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)  # vcomp140.dll is not redistributed with packs
set(GGML_SSE42 ON CACHE BOOL "" FORCE)
set(GGML_AVX ${_sigil_avx} CACHE BOOL "" FORCE)
set(GGML_AVX2 ${_sigil_avx2} CACHE BOOL "" FORCE)
set(GGML_FMA ${_sigil_avx2} CACHE BOOL "" FORCE)
set(GGML_F16C ${_sigil_avx2} CACHE BOOL "" FORCE)
set(GGML_BMI2 ${_sigil_avx2} CACHE BOOL "" FORCE)
set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512 ${_sigil_avx512} CACHE BOOL "" FORCE)
set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)
set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)

# --- GPU backends ---------------------------------------------------------------------------------
set(SIGIL_GPU_BACKEND "none")
if(SIGIL_VARIANT STREQUAL "vulkan")
  set(SIGIL_GPU_BACKEND "vulkan")
  set(GGML_VULKAN ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  find_package(Vulkan)
  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "The vulkan variant needs the Vulkan SDK (set VULKAN_SDK); see scripts/native/README.md")
  endif()
elseif(SIGIL_VARIANT STREQUAL "cuda13")
  set(SIGIL_GPU_BACKEND "cuda")
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  set(SIGIL_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
      CACHE STRING "CUDA architectures of the cuda13 variant (Turing and newer)")
  set(CMAKE_CUDA_ARCHITECTURES "${SIGIL_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
  find_package(CUDAToolkit 13.0)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "The cuda13 variant needs the CUDA 13 toolkit (set CUDAToolkit_ROOT); see scripts/native/README.md")
  endif()
  if(CUDAToolkit_VERSION_MAJOR VERSION_LESS 13)
    message(FATAL_ERROR "The cuda13 variant needs CUDA 13 (found ${CUDAToolkit_VERSION})")
  endif()
else()
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()

message(STATUS "Sigil variant: ${SIGIL_VARIANT} (CPU ${SIGIL_VARIANT_ARCH_FLAG}, GPU ${SIGIL_GPU_BACKEND}, CRT /MD)")
