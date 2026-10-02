
# ---------------------------------------------------------------------------------------------
# Sigil overlay (scripts/runtime-packs/qwentts/server/CMakeLists.append.cmake, appended by
# build.ps1 to the pinned qwentts.cpp CMakeLists.txt): the sigil-tts-server helper of the
# runtime/qwentts-server-* packs. Same libraries and options as upstream's tts-server target.
# ---------------------------------------------------------------------------------------------
set(SIGIL_QWENTTS_COMMIT "unknown" CACHE STRING "qwentts.cpp commit the pack is built from")
set(SIGIL_PACK_VERSION "dev" CACHE STRING "Catalog version of the runtime pack")
set(SIGIL_PACK_VARIANT "unknown" CACHE STRING "Runtime pack variant (cpu, vulkan, cuda13)")
add_executable(sigil-tts-server tools/sigil-tts-server.cpp)
target_link_libraries(sigil-tts-server PRIVATE qwen-core httplib yyjson)
target_compile_definitions(sigil-tts-server PRIVATE
    SIGIL_QWENTTS_COMMIT="${SIGIL_QWENTTS_COMMIT}"
    SIGIL_PACK_VERSION="${SIGIL_PACK_VERSION}"
    SIGIL_PACK_VARIANT="${SIGIL_PACK_VARIANT}")
link_ggml_backends(sigil-tts-server)
