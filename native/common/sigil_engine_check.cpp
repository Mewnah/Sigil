// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_engine_check - validate a built sigil_engine DLL the way Sigil loads it, and print its
// description as JSON (used by scripts/native/build-runtime-packs.ps1 to write sigil-pack.json).
//
//   sigil_engine_check <path\to\sigil_<engine>.dll> [--config <json>] [--variant <variant>]
//                      [--dll-dir <folder>]... [--no-create]
//
// Loads the DLL by full path with LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
// LOAD_LIBRARY_SEARCH_DEFAULT_DIRS), calls sigil_engine_abi_version() first, resolves all 16 ABI v1
// exports, validates sigil_engine_get_info(), lists the DLL's imports (the build script refuses
// debug CRTs), then creates an instance with the config (default {}), runs {"op":"info"} and destroys
// it. Prints one JSON object on stdout; exit code 0 when every check passed, 1 otherwise, 2 on usage
// errors. Engine log lines go to stderr.
//
//   --dll-dir <folder>  AddDllDirectory before the load (the same search as Sigil after it registers
//                       a runtime pack's folder): e.g. the CUDA toolkit's bin\x64 for a cuda13 engine
//                       linked with the dynamic cudart / cuBLAS. Repeatable.
//   --variant <v>       The variant the DLL was built as (cpu, cpu-avx2, cpu-avx512, vulkan, cuda13).
//                       When this PC's CPU lacks the variant's instruction sets, create is skipped
//                       (its code would die with an illegal instruction) and only the structural
//                       checks run; for a GPU variant, a create that fails with
//                       SIGIL_E_BACKEND_UNAVAILABLE (no such GPU on the build PC) is reported as
//                       skipped instead of failed.
//   --no-create         Structural checks only (load, ABI, exports, info, imports).
//
// This tool is built without the variant's /arch flags (common/CMakeLists.txt), so it runs on any
// x64 build PC whatever variant tree it belongs to.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "sigil_engine.h"
#include "sigil_json.hpp"

namespace {

std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n, nullptr, nullptr);
    return out;
}

std::string last_win32_error() {
    const DWORD code = GetLastError();
    wchar_t* buf = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0,
                   reinterpret_cast<wchar_t*>(&buf), 0, nullptr);
    std::wstring msg = buf ? buf : L"";
    if (buf) LocalFree(buf);
    while (!msg.empty() && (msg.back() == L'\n' || msg.back() == L'\r' || msg.back() == L' ')) msg.pop_back();
    return utf8(msg) + " (error " + std::to_string(code) + ")";
}

std::wstring full_path(const wchar_t* p) {
    wchar_t full[MAX_PATH * 4];
    const DWORD n = GetFullPathNameW(p, static_cast<DWORD>(std::size(full)), full, nullptr);
    return (n > 0 && n < std::size(full)) ? std::wstring(full, n) : std::wstring(p);
}

void SIGIL_ENGINE_CALL log_to_stderr(void*, int32_t level, const char* message) {
    static const char* names[] = {"trace", "debug", "info", "warn", "error"};
    const char* name = (level >= 0 && level <= 4) ? names[level] : "error";
    std::fprintf(stderr, "[engine %s] %s\n", name, message ? message : "");
}

const char* const kSymbols[16] = {
    "sigil_engine_abi_version", "sigil_engine_get_info", "sigil_engine_last_error", "sigil_engine_set_log_callback",
    "sigil_engine_create",      "sigil_engine_destroy",  "sigil_engine_cancel",     "sigil_engine_query",
    "sigil_engine_process",     "sigil_engine_render",   "sigil_stream_open",       "sigil_stream_push",
    "sigil_stream_finish",      "sigil_stream_close",    "sigil_free_string",       "sigil_free_audio",
};
const char* const kVariants[] = {"cpu", "cpu-avx2", "cpu-avx512", "vulkan", "cuda13"};

using AbiVersionFn = uint32_t(SIGIL_ENGINE_CALL*)(void);
using GetInfoFn = const sigil_engine_info*(SIGIL_ENGINE_CALL*)(void);
using LastErrorFn = const char*(SIGIL_ENGINE_CALL*)(void);
using SetLogFn = void(SIGIL_ENGINE_CALL*)(sigil_log_fn, void*);
using CreateFn = sigil_status(SIGIL_ENGINE_CALL*)(const char*, sigil_engine**);
using DestroyFn = void(SIGIL_ENGINE_CALL*)(sigil_engine*);
using QueryFn = sigil_status(SIGIL_ENGINE_CALL*)(sigil_engine*, const char*, char**);
using FreeStringFn = void(SIGIL_ENGINE_CALL*)(char*);

template <class T>
T sym(HMODULE m, const char* name) {
    return reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(m, name)));
}

int finish(sigil::json::Writer& w, bool ok) {
    w.field("ok", ok).end_object();
    std::fputs(w.str().c_str(), stdout);
    std::fputc('\n', stdout);
    return ok ? 0 : 1;
}

// --- CPU features (the same sets native_runtime::VariantSupport checks) --------------------------

struct CpuFeatures {
    bool avx2 = false;    // AVX2 + FMA + F16C + BMI2, with the OS saving the YMM state
    bool avx512 = false;  // AVX-512 F/CD/BW/DQ/VL, with the OS saving the ZMM state
};

CpuFeatures detect_cpu() {
    CpuFeatures f;
    int r[4] = {};
    __cpuid(r, 0);
    const int max_leaf = r[0];
    if (max_leaf < 7) return f;
    __cpuid(r, 1);
    const bool osxsave = (r[2] >> 27) & 1;
    const bool fma = (r[2] >> 12) & 1;
    const bool f16c = (r[2] >> 29) & 1;
    const bool avx = (r[2] >> 28) & 1;
    if (!osxsave || !avx) return f;
    const unsigned long long xcr0 = _xgetbv(0);
    const bool ymm_state = (xcr0 & 0x6) == 0x6;
    const bool zmm_state = (xcr0 & 0xE6) == 0xE6;
    __cpuidex(r, 7, 0);
    const unsigned ebx = static_cast<unsigned>(r[1]);
    const bool avx2 = (ebx >> 5) & 1;
    const bool bmi2 = (ebx >> 8) & 1;
    f.avx2 = ymm_state && avx2 && fma && f16c && bmi2;
    const bool avx512 = ((ebx >> 16) & 1) && ((ebx >> 17) & 1) && ((ebx >> 28) & 1) && ((ebx >> 30) & 1) && ((ebx >> 31) & 1);
    f.avx512 = f.avx2 && zmm_state && avx512;
    return f;
}

/// Why this PC's CPU cannot execute `variant`'s code (empty when it can).
std::string cpu_gap(const std::string& variant, const CpuFeatures& cpu) {
    if (variant == "cpu-avx512" && !cpu.avx512) return "this build PC's CPU does not support AVX-512";
    if ((variant == "cpu-avx2" || variant == "vulkan" || variant == "cuda13") && !cpu.avx2) {
        return "this build PC's CPU does not support AVX2";
    }
    return {};
}

bool is_gpu_variant(const std::string& v) { return v == "vulkan" || v == "cuda13"; }

// --- Imports of the loaded module (import and delay-import tables) --------------------------------

std::vector<std::string> module_imports(HMODULE m) {
    std::vector<std::string> out;
    const auto* base = reinterpret_cast<const unsigned char*>(m);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return out;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return out;
    const auto& dirs = nt->OptionalHeader.DataDirectory;
    auto add = [&](DWORD name_rva) {
        if (!name_rva || name_rva >= nt->OptionalHeader.SizeOfImage) return;
        std::string name(reinterpret_cast<const char*>(base + name_rva));
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(std::move(name));
    };
    const auto& imp = dirs[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (imp.VirtualAddress && imp.Size) {
        for (const auto* d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + imp.VirtualAddress); d->Name; ++d) add(d->Name);
    }
    const auto& delay = dirs[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
    if (delay.VirtualAddress && delay.Size) {
        for (const auto* d = reinterpret_cast<const IMAGE_DELAYLOAD_DESCRIPTOR*>(base + delay.VirtualAddress); d->DllNameRVA; ++d) {
            add(d->DllNameRVA);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) {
        std::fputs(
            "usage: sigil_engine_check <path\\to\\engine.dll> [--config <json>] [--variant <v>] [--dll-dir <folder>]... "
            "[--no-create]\n",
            stderr);
        return 2;
    }
    std::string config = "{}";
    std::string variant;
    std::vector<std::wstring> dll_dirs;
    bool no_create = false;
    for (int i = 2; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--config" && i + 1 < argc) {
            config = utf8(argv[++i]);
        } else if (a == L"--variant" && i + 1 < argc) {
            variant = utf8(argv[++i]);
            if (std::find_if(std::begin(kVariants), std::end(kVariants), [&](const char* v) { return variant == v; }) ==
                std::end(kVariants)) {
                std::fprintf(stderr, "unknown variant: %s\n", variant.c_str());
                return 2;
            }
        } else if (a == L"--dll-dir" && i + 1 < argc) {
            dll_dirs.push_back(full_path(argv[++i]));
        } else if (a == L"--no-create") {
            no_create = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", utf8(a).c_str());
            return 2;
        }
    }

    const std::wstring path = full_path(argv[1]);
    const CpuFeatures cpu = detect_cpu();

    sigil::json::Writer w;
    w.begin_object().field("path", utf8(path)).field("host_abi_version", SIGIL_ENGINE_ABI_VERSION);
    if (!variant.empty()) w.field("variant", variant);
    w.key("build_cpu").begin_object().field("avx2", cpu.avx2).field("avx512", cpu.avx512).end_object();

    w.key("dll_dirs").begin_array();
    for (const auto& d : dll_dirs) w.value(utf8(d));
    w.end_array();
    for (const auto& d : dll_dirs) {
        const DWORD attrs = GetFileAttributesW(d.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            w.field("error", "--dll-dir " + utf8(d) + " is not a folder");
            return finish(w, false);
        }
        if (!AddDllDirectory(d.c_str())) {
            w.field("error", "AddDllDirectory(" + utf8(d) + ") failed: " + last_win32_error());
            return finish(w, false);
        }
    }

    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!m) {
        w.field("error", "cannot load the DLL: " + last_win32_error() +
                             (dll_dirs.empty() ? "" : " (a dependency may be missing from the --dll-dir folders)"));
        return finish(w, false);
    }

    w.key("imports").begin_array();
    for (const auto& name : module_imports(m)) w.value(name);
    w.end_array();

    const auto abi = sym<AbiVersionFn>(m, "sigil_engine_abi_version");
    if (!abi) {
        w.field("error", "not a Sigil engine: sigil_engine_abi_version is not exported");
        return finish(w, false);
    }
    const uint32_t version = abi();
    w.field("abi_version", version);
    if (version != SIGIL_ENGINE_ABI_VERSION) {
        w.field("error", "ABI v" + std::to_string(version) + ", but this checker validates v" + std::to_string(SIGIL_ENGINE_ABI_VERSION));
        return finish(w, false);
    }

    std::vector<std::string> missing;
    for (const char* s : kSymbols) {
        if (!GetProcAddress(m, s)) missing.emplace_back(s);
    }
    w.key("missing_exports").begin_array();
    for (const auto& s : missing) w.value(s);
    w.end_array();
    if (!missing.empty()) {
        w.field("error", "missing exports");
        return finish(w, false);
    }

    const sigil_engine_info* info = sym<GetInfoFn>(m, "sigil_engine_get_info")();
    if (!info) {
        w.field("error", "sigil_engine_get_info returned NULL");
        return finish(w, false);
    }
    if (info->struct_size < 56 || info->abi_version != SIGIL_ENGINE_ABI_VERSION || !info->name || !info->version || !info->variant ||
        !info->kind || !info->build_info) {
        w.field("error", "sigil_engine_get_info returned an invalid description");
        return finish(w, false);
    }
    w.key("info")
        .begin_object()
        .field("struct_size", info->struct_size)
        .field("abi_version", info->abi_version)
        .field("name", info->name)
        .field("version", info->version)
        .field("variant", info->variant)
        .field("kind", info->kind)
        .field("build_info", info->build_info)
        .field("capabilities", static_cast<uint64_t>(info->capabilities))
        .end_object();

    // Instruction sets: the variant it was built as and the DLL's own variant (what Sigil checks).
    const std::string built = variant.empty() ? std::string(info->variant) : variant;
    std::string skip_create;
    if (no_create) {
        skip_create = "--no-create";
    } else {
        skip_create = cpu_gap(built, cpu);
        if (skip_create.empty()) skip_create = cpu_gap(info->variant, cpu);
    }

    sym<SetLogFn>(m, "sigil_engine_set_log_callback")(log_to_stderr, nullptr);
    const auto last_error = sym<LastErrorFn>(m, "sigil_engine_last_error");
    const auto create = sym<CreateFn>(m, "sigil_engine_create");
    const auto destroy = sym<DestroyFn>(m, "sigil_engine_destroy");
    const auto query = sym<QueryFn>(m, "sigil_engine_query");
    const auto free_string = sym<FreeStringFn>(m, "sigil_free_string");

    bool ok = true;
    w.key("create").begin_object();
    if (!skip_create.empty()) {
        w.field("skipped", skip_create);
    } else {
        sigil_engine* engine = nullptr;
        const sigil_status st = create(config.c_str(), &engine);
        w.field("status", st);
        if (st != SIGIL_OK || !engine) {
            const char* e = last_error();
            w.field("error", e ? e : "");
            if (st == SIGIL_E_BACKEND_UNAVAILABLE && is_gpu_variant(built)) {
                // No such GPU (or driver) on the build PC: the structural checks passed.
                w.field("skipped", "the " + built + " backend is not available on this build PC");
            } else {
                ok = false;
            }
        } else {
            char* out = nullptr;
            const sigil_status qs = query(engine, "{\"op\":\"info\"}", &out);
            w.field("query_status", qs);
            if (qs == SIGIL_OK && out) {
                std::string parsed;
                std::string problem;
                try {
                    parsed = sigil::json::Value::parse(out).dump();
                } catch (const std::exception& e) {
                    problem = std::string("the info query returned invalid JSON: ") + e.what();
                }
                if (problem.empty()) {
                    w.key("query_info").raw(parsed);
                } else {
                    ok = false;
                    w.field("query_error", problem);
                }
            } else {
                ok = false;
                const char* e = last_error();
                w.field("query_error", e ? e : "");
            }
            free_string(out);
            destroy(engine);
        }
    }
    w.end_object();
    sym<SetLogFn>(m, "sigil_engine_set_log_callback")(nullptr, nullptr);
    FreeLibrary(m);
    return finish(w, ok);
}
