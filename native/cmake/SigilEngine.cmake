# SPDX-License-Identifier: AGPL-3.0-only
#
# SigilEngine.cmake - sigil_add_engine(): one sigil_engine ABI v1 DLL.
#
#   sigil_add_engine(<name>
#     VERSION <major.minor.patch[-suffix]>
#     SOURCES <files...>
#     [LIBRARIES <targets...>]        # linked PRIVATE (static third-party libraries, e.g. ggml)
#     [INCLUDES <dirs...>])
#
# Produces sigil_<name>.dll (unique name per engine: the Windows loader matches DLLs by base name),
# with:
# - the shared ABI implementation (common/sigil_engine_impl.cpp, sigil_json.cpp) compiled in and
#   SIGIL_ENGINE_BUILD defined, so exactly the 16 sigil_engine exports are exported;
# - SIGIL_ENGINE_VERSION, SIGIL_ENGINE_VARIANT and SIGIL_ENGINE_BUILD_INFO string macros for the
#   engine's sigil::engine_descriptor();
# - a VERSIONINFO resource (FileVersion = VERSION), /W4 for the engine's own sources, /guard:cf,
#   a PDB with a relative path, /OPT:REF,ICF;
# - the variant settings of SigilVariant.cmake (/MD, instruction sets).

set(SIGIL_ENGINE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")

function(sigil_add_engine name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "VERSION" "SOURCES;LIBRARIES;INCLUDES")
  if(NOT name MATCHES "^[a-z0-9][a-z0-9_-]*$")
    message(FATAL_ERROR "sigil_add_engine: invalid engine name '${name}' (lower-case letters, digits, '-', '_')")
  endif()
  if(NOT ARG_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    message(FATAL_ERROR "sigil_add_engine(${name}): VERSION must start with major.minor.patch (got '${ARG_VERSION}')")
  endif()
  set(_major ${CMAKE_MATCH_1})
  set(_minor ${CMAKE_MATCH_2})
  set(_patch ${CMAKE_MATCH_3})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "sigil_add_engine(${name}): SOURCES is empty")
  endif()

  set(target sigil_${name})
  add_library(${target} SHARED ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE sigil::engine_abi ${ARG_LIBRARIES})
  if(ARG_INCLUDES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDES})
  endif()

  # No ';' in the build info: it would split the definition list.
  set(_build_info "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, ${SIGIL_VARIANT}, CPU ${SIGIL_VARIANT_ARCH_FLAG}, GPU ${SIGIL_GPU_BACKEND}, /MD, $<CONFIG>")
  target_compile_definitions(${target} PRIVATE
    "SIGIL_ENGINE_VERSION=\"${ARG_VERSION}\""
    "SIGIL_ENGINE_VARIANT=\"${SIGIL_VARIANT}\""
    "SIGIL_ENGINE_BUILD_INFO=\"${_build_info}\"")

  set(_rc "${CMAKE_CURRENT_BINARY_DIR}/${target}_version.rc")
  set(SIGIL_RC_NAME "${name}")
  set(SIGIL_RC_TARGET "${target}")
  set(SIGIL_RC_VERSION "${ARG_VERSION}")
  set(SIGIL_RC_MAJOR ${_major})
  set(SIGIL_RC_MINOR ${_minor})
  set(SIGIL_RC_PATCH ${_patch})
  set(SIGIL_RC_VARIANT "${SIGIL_VARIANT}")
  configure_file("${SIGIL_ENGINE_CMAKE_DIR}/sigil_engine_version.rc.in" "${_rc}" @ONLY)
  if(WIN32)
    target_sources(${target} PRIVATE "${_rc}")
  endif()

  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${target}"
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

  if(MSVC)
    # Warnings only for Sigil's own sources (third-party libraries keep their own flags).
    target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:__cplusplus /EHsc /guard:cf /sdl)
    target_link_options(${target} PRIVATE /DEBUG /OPT:REF /OPT:ICF /guard:cf /DYNAMICBASE /HIGHENTROPYVA
      /NXCOMPAT "/PDBALTPATH:%_PDB%")
  endif()

  set_property(GLOBAL APPEND PROPERTY SIGIL_ENGINE_TARGETS ${target})
  install(TARGETS ${target} RUNTIME DESTINATION . LIBRARY DESTINATION .)
  sigil_record_output(${target})
endfunction()

# sigil_record_output(<target>): write the target's exact output path per configuration to
# <build>/sigil-outputs/<config>/<target>.path. scripts/native/build-runtime-packs.ps1 reads it
# instead of searching the tree: a multi-config (Visual Studio) tree keeps Debug/ and Release/ side by
# side, and "the newest DLL" can be a stale build of the other configuration.
function(sigil_record_output target)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sigil-outputs/$<CONFIG>/${target}.path"
       CONTENT "$<TARGET_FILE:${target}>")
endfunction()
