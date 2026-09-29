# SPDX-License-Identifier: AGPL-3.0-only
#
# sigil_transcribe.dll: transcribe.cpp (pinned in UPSTREAM.json) behind the sigil_engine ABI v1.
# Owned by P1-G2 (docs/adr/0003b-transcribe-cpp.md). Version and variants: engine.json.
#
# Built like every Sigil engine (native/cmake/SigilVariant.cmake + SigilEngine.cmake, the shared ABI
# implementation in native/common): transcribe.cpp and its vendored ggml are linked STATICALLY into
# the uniquely named DLL (no ggml*.dll another in-process engine could bind by base name), /MD,
# GGML_NATIVE=OFF, GGML_BACKEND_DL=OFF, the variant's instruction sets, exactly the 16 exports.
#
# Two ways in:
#   - as part of native/ (scripts/native/build-runtime-packs.ps1 -Engine transcribe -Variant ...,
#     or cmake -S native -DSIGIL_VARIANT=... -DSIGIL_ENGINES=transcribe);
#   - standalone (native/engines/transcribe/build.ps1, which also fetches the pinned source, adds the
#     licenses to the pack and runs the self-test): cmake -S native/engines/transcribe -DSIGIL_VARIANT=...
# The transcribe.cpp source comes from TRANSCRIBE_SOURCE_DIR, else from the shared dev cache that
# build.ps1 fills (%LOCALAPPDATA%\sigil-dev-cache\transcribe.cpp\src-<commit>); with
# -DSIGIL_TRANSCRIBE_FETCH=ON configure downloads and verifies the pinned archive itself.

cmake_minimum_required(VERSION 3.25)

if(NOT COMMAND sigil_add_engine)
  # Standalone: this engine alone, with the same rules and ABI implementation as native/.
  project(sigil_transcribe LANGUAGES C CXX)
  get_filename_component(_sigil_native "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${_sigil_native}/cmake")
  set(SIGIL_NATIVE_DIR "${_sigil_native}" CACHE INTERNAL "")
  include(SigilVariant)
  include(SigilEngine)
  add_subdirectory("${_sigil_native}/common" "${CMAKE_CURRENT_BINARY_DIR}/sigil-common")
  set(_sigil_transcribe_standalone ON)
else()
  set(_sigil_transcribe_standalone OFF)
endif()

option(SIGIL_TRANSCRIBE_SELFTEST "Build sigil_transcribe_selftest.exe (ABI contract + real-model smoke run)" ON)
option(SIGIL_TRANSCRIBE_CLI "Also build upstream transcribe-cli.exe (reference outputs)" OFF)
option(SIGIL_TRANSCRIBE_FETCH "Download the pinned transcribe.cpp archive at configure time when it is not in the dev cache" OFF)
set(TRANSCRIBE_SOURCE_DIR "" CACHE PATH "transcribe.cpp source tree at the commit pinned in UPSTREAM.json (empty: the dev cache)")

# ---------------------------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------------------------

file(READ "${CMAKE_CURRENT_LIST_DIR}/engine.json" _tr_engine_json)
string(JSON _tr_version GET "${_tr_engine_json}" version)
file(READ "${CMAKE_CURRENT_LIST_DIR}/UPSTREAM.json" _tr_up_json)
string(JSON _tr_up_version GET "${_tr_up_json}" version)
string(JSON _tr_up_commit GET "${_tr_up_json}" commit)
string(JSON _tr_up_abihash GET "${_tr_up_json}" abihash)
string(JSON _tr_up_ggml GET "${_tr_up_json}" ggml_commit)
string(JSON _tr_shim_rev GET "${_tr_up_json}" shim_revision)
string(JSON _tr_cuda_archs GET "${_tr_up_json}" cuda architectures)
string(JSON _tr_archive_url GET "${_tr_up_json}" archive url)
string(JSON _tr_archive_file GET "${_tr_up_json}" archive file)
string(JSON _tr_archive_sha GET "${_tr_up_json}" archive sha256)
string(JSON _tr_archive_root GET "${_tr_up_json}" archive root)
string(SUBSTRING "${_tr_up_commit}" 0 7 _tr_up_short)
string(SUBSTRING "${_tr_up_ggml}" 0 7 _tr_ggml_short)
if(NOT _tr_version STREQUAL "${_tr_up_version}-s${_tr_shim_rev}")
  message(FATAL_ERROR "engine.json version '${_tr_version}' must be '${_tr_up_version}-s${_tr_shim_rev}' "
                      "(UPSTREAM.json version and shim_revision)")
endif()

# ---------------------------------------------------------------------------------------------
# Pinned source
# ---------------------------------------------------------------------------------------------

if(NOT TRANSCRIBE_SOURCE_DIR)
  if(DEFINED ENV{SIGIL_DEV_CACHE} AND NOT "$ENV{SIGIL_DEV_CACHE}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{SIGIL_DEV_CACHE}/transcribe.cpp" _tr_cache)
  else()
    file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}/sigil-dev-cache/transcribe.cpp" _tr_cache)
  endif()
  set(_tr_tree "${_tr_cache}/src-${_tr_up_short}")
  set(_tr_marker "${_tr_tree}/.sigil-extracted")
  set(_tr_ready OFF)
  if(EXISTS "${_tr_marker}")
    file(READ "${_tr_marker}" _tr_marker_sha)
    string(STRIP "${_tr_marker_sha}" _tr_marker_sha)
    if(_tr_marker_sha STREQUAL _tr_archive_sha)
      set(_tr_ready ON)
    endif()
  endif()
  if(NOT _tr_ready AND SIGIL_TRANSCRIBE_FETCH)
    set(_tr_archive "${_tr_cache}/${_tr_archive_file}")
    if(NOT EXISTS "${_tr_archive}")
      message(STATUS "Downloading transcribe.cpp ${_tr_up_version} (${_tr_up_short}) from ${_tr_archive_url}")
      file(DOWNLOAD "${_tr_archive_url}" "${_tr_archive}.partial" EXPECTED_HASH SHA256=${_tr_archive_sha}
           TLS_VERIFY ON STATUS _tr_dl)
      list(GET _tr_dl 0 _tr_dl_code)
      if(NOT _tr_dl_code EQUAL 0)
        file(REMOVE "${_tr_archive}.partial")
        message(FATAL_ERROR "transcribe.cpp download failed: ${_tr_dl}")
      endif()
      file(RENAME "${_tr_archive}.partial" "${_tr_archive}")
    endif()
    file(SHA256 "${_tr_archive}" _tr_have_sha)
    if(NOT _tr_have_sha STREQUAL _tr_archive_sha)
      message(FATAL_ERROR "${_tr_archive} has sha256 ${_tr_have_sha}, UPSTREAM.json pins ${_tr_archive_sha}")
    endif()
    set(_tr_staging "${_tr_tree}.staging")
    file(REMOVE_RECURSE "${_tr_staging}" "${_tr_tree}")
    file(MAKE_DIRECTORY "${_tr_staging}")
    # System32 tar (bsdtar); upstream's agent / CI folders are not needed (ggml/.pi holds a symlink
    # Windows cannot create without Developer Mode).
    execute_process(
      COMMAND "$ENV{SystemRoot}/System32/tar.exe" -xzf "${_tr_archive}" -C "${_tr_staging}"
              --exclude "*/.claude" --exclude "*/.pi" --exclude "*/.github" --exclude "*/.cargo"
      RESULT_VARIABLE _tr_tar_rc)
    if(NOT _tr_tar_rc EQUAL 0)
      message(FATAL_ERROR "extracting ${_tr_archive} failed (${_tr_tar_rc})")
    endif()
    file(RENAME "${_tr_staging}" "${_tr_tree}")
    file(WRITE "${_tr_marker}" "${_tr_archive_sha}")
    set(_tr_ready ON)
  endif()
  if(NOT _tr_ready)
    if(NOT _sigil_transcribe_standalone AND NOT SIGIL_ENGINES)
      # Building every engine of native/ without the transcribe source: skip this one, don't fail.
      message(WARNING "sigil_transcribe skipped: the pinned transcribe.cpp source is not in ${_tr_cache} "
                      "(run native/engines/transcribe/build.ps1 -FetchOnly, or configure with -DSIGIL_TRANSCRIBE_FETCH=ON)")
      return()
    endif()
    message(FATAL_ERROR "The pinned transcribe.cpp ${_tr_up_version} source is not in ${_tr_cache}. "
                        "Run `pwsh native/engines/transcribe/build.ps1 -FetchOnly` (one ~44 MB download, sha256-checked), "
                        "configure with -DSIGIL_TRANSCRIBE_FETCH=ON, or set TRANSCRIBE_SOURCE_DIR.")
  endif()
  set(TRANSCRIBE_SOURCE_DIR "${_tr_tree}/${_tr_archive_root}" CACHE PATH "" FORCE)
endif()

if(NOT EXISTS "${TRANSCRIBE_SOURCE_DIR}/include/transcribe.h")
  message(FATAL_ERROR "TRANSCRIBE_SOURCE_DIR (${TRANSCRIBE_SOURCE_DIR}) is not a transcribe.cpp source tree")
endif()
file(READ "${TRANSCRIBE_SOURCE_DIR}/include/transcribe.h" _tr_header)
string(REGEX MATCH "define +TRANSCRIBE_VERSION_MAJOR +([0-9]+)" _ "${_tr_header}")
set(_tr_found "${CMAKE_MATCH_1}")
string(REGEX MATCH "define +TRANSCRIBE_VERSION_MINOR +([0-9]+)" _ "${_tr_header}")
string(APPEND _tr_found ".${CMAKE_MATCH_1}")
string(REGEX MATCH "define +TRANSCRIBE_VERSION_PATCH +([0-9]+)" _ "${_tr_header}")
string(APPEND _tr_found ".${CMAKE_MATCH_1}")
if(NOT _tr_found STREQUAL _tr_up_version)
  message(FATAL_ERROR "TRANSCRIBE_SOURCE_DIR holds transcribe.cpp ${_tr_found}; UPSTREAM.json pins ${_tr_up_version}")
endif()
if(EXISTS "${TRANSCRIBE_SOURCE_DIR}/include/transcribe.abihash")
  file(READ "${TRANSCRIBE_SOURCE_DIR}/include/transcribe.abihash" _tr_found_abihash)
  string(STRIP "${_tr_found_abihash}" _tr_found_abihash)
  if(NOT _tr_found_abihash STREQUAL _tr_up_abihash)
    message(FATAL_ERROR "transcribe.h ABI hash ${_tr_found_abihash} != pinned ${_tr_up_abihash}: "
                        "the C API changed; review shim.cpp before bumping UPSTREAM.json")
  endif()
endif()

# ---------------------------------------------------------------------------------------------
# transcribe.cpp as a static library (the variant's ggml options come from SigilVariant.cmake)
# ---------------------------------------------------------------------------------------------

set(TRANSCRIBE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_EXAMPLES ${SIGIL_TRANSCRIBE_CLI} CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_INSTALL OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_USE_OPENMP OFF CACHE BOOL "" FORCE)      # ggml's own thread pool
set(TRANSCRIBE_USE_SYSTEM_BLAS OFF CACHE BOOL "" FORCE) # nothing that could collide in Sigil's process
set(TRANSCRIBE_GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_ENABLE_VALIDATION_HOOKS OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_METAL OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_HIP OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_X86_CONSERVATIVE OFF CACHE BOOL "" FORCE)  # SigilVariant.cmake sets the SIMD floor
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)

set(_tr_gpu 0)
set(_tr_default_backend "cpu")
set(_tr_gpu_info "")
set(TRANSCRIBE_CUDA OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_VULKAN OFF CACHE BOOL "" FORCE)
# The pre-override cache variable of older build trees held the effective list itself and silently
# outlived a development build; the effective list is now computed below on every configure.
unset(SIGIL_TRANSCRIBE_CUDA_ARCHITECTURES CACHE)
set(SIGIL_TRANSCRIBE_CUDA_ARCHS_OVERRIDE "" CACHE STRING
    "Development only: CUDA architectures of sigil_transcribe (cuda13) instead of UPSTREAM.json's pinned list (empty = the pin). build.ps1 sets it on every configure; packs built with it are marked -dev.")
if(SIGIL_VARIANT STREQUAL "cuda13")
  set(TRANSCRIBE_CUDA ON CACHE BOOL "" FORCE)  # upstream maps TRANSCRIBE_CUDA onto GGML_CUDA
  # This engine's own GPU generations (ggml-cuda compiles every kernel per architecture): Turing,
  # Ampere, Ada and Blackwell as native code plus compute_80 PTX for anything else. A normal
  # variable (not the cache), so a later edit of the pin always takes effect.
  if(SIGIL_TRANSCRIBE_CUDA_ARCHS_OVERRIDE)
    set(_tr_effective_archs "${SIGIL_TRANSCRIBE_CUDA_ARCHS_OVERRIDE}")
    if(NOT _tr_effective_archs STREQUAL _tr_cuda_archs)
      message(WARNING "sigil_transcribe (cuda13) is built for ${_tr_effective_archs} only (development override; "
                      "UPSTREAM.json pins ${_tr_cuda_archs}). Do not ship this DLL.")
    endif()
  else()
    set(_tr_effective_archs "${_tr_cuda_archs}")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_tr_effective_archs}")
  set(_tr_gpu 1)
  set(_tr_default_backend "cuda")
  string(REPLACE ";" " " _tr_archs_text "${CMAKE_CUDA_ARCHITECTURES}")
  set(_tr_gpu_info ", CUDA ${CUDAToolkit_VERSION} archs ${_tr_archs_text}")
  if(NOT _tr_effective_archs STREQUAL _tr_cuda_archs)
    string(APPEND _tr_gpu_info " (dev override)")
  endif()
elseif(SIGIL_VARIANT STREQUAL "vulkan")
  set(TRANSCRIBE_VULKAN ON CACHE BOOL "" FORCE)
  set(_tr_gpu 1)
  set(_tr_default_backend "vulkan")
endif()

if(MSVC)
  # native/cmake adds a few MSVC flags with add_compile_options; keep them away from nvcc.
  get_directory_property(_tr_opts COMPILE_OPTIONS)
  set(_tr_guarded "")
  foreach(_o IN LISTS _tr_opts)
    if(_o MATCHES "^\\$<")
      list(APPEND _tr_guarded "${_o}")
    else()
      list(APPEND _tr_guarded "$<$<COMPILE_LANGUAGE:C,CXX>:${_o}>")
    endif()
  endforeach()
  set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_tr_guarded}")
endif()

add_subdirectory("${TRANSCRIBE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/transcribe" EXCLUDE_FROM_ALL)
# Upstream publishes ${CMAKE_SOURCE_DIR}/include, which is only right when it is the top-level project.
target_include_directories(transcribe PUBLIC "${TRANSCRIBE_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------------------------
# sigil_transcribe.dll
# ---------------------------------------------------------------------------------------------

sigil_add_engine(transcribe
  VERSION "${_tr_version}"
  SOURCES shim.cpp
  LIBRARIES transcribe)
# shim.cpp calls ggml_set_abort_callback on the DLL's own static ggml (linked through transcribe;
# upstream links ggml PRIVATE, so its headers are not propagated).
target_include_directories(sigil_transcribe PRIVATE "${TRANSCRIBE_SOURCE_DIR}/ggml/include")
target_compile_definitions(sigil_transcribe PRIVATE
  "SIGIL_TRANSCRIBE_DEFAULT_BACKEND=\"${_tr_default_backend}\""
  "SIGIL_TRANSCRIBE_GPU=${_tr_gpu}"
  "SIGIL_TRANSCRIBE_UPSTREAM_INFO=\"transcribe.cpp ${_tr_up_version} (${_tr_up_short}), ggml ${_tr_ggml_short}, static ggml${_tr_gpu_info}\"")
if(_sigil_transcribe_standalone)
  set_target_properties(sigil_transcribe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/out"
    PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/out")
endif()

if(SIGIL_TRANSCRIBE_SELFTEST AND WIN32)
  # Loads the DLL at run time like Sigil does; never links it.
  add_executable(sigil_transcribe_selftest selftest.cpp)
  target_include_directories(sigil_transcribe_selftest PRIVATE "${SIGIL_NATIVE_DIR}/common")
  target_compile_features(sigil_transcribe_selftest PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(sigil_transcribe_selftest PRIVATE /W4 /permissive- /EHsc)
  endif()
  set_target_properties(sigil_transcribe_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tools")
  add_dependencies(sigil_transcribe_selftest sigil_transcribe)
endif()

message(STATUS "sigil_transcribe ${_tr_version} (${SIGIL_VARIANT}, default backend ${_tr_default_backend}) from ${TRANSCRIBE_SOURCE_DIR}")
