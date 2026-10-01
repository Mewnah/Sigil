// sigil-bergamot: Mozilla's Bergamot translation engine behind a JSON-lines stdio protocol, run by
// Sigil's process host (src-tauri/src/services/process_host, recipe `bergamot`) and driven by
// src-tauri/src/services/mt_local/bergamot/client.rs. Protocol 1 is specified in
// docs/adr/0008-bergamot.md and mt_local/bergamot/protocol.rs.
//
//   sigil-bergamot.exe --jsonl [--threads N] [--cache-size N]
//
// stdin:  one JSON request per line: {"id":N,"op":"hello"|"load"|"unload"|"translate"|"cancel"|"stats",...}
// stdout: one JSON reply per line:   {"id":N,"ok":true,"result":{...}} | {"id":N,"ok":false,"error":{"kind","message"}}
// stderr: diagnostics only (the process host logs them). EOF on stdin = quit.
//
// Everything the upstream libraries might print to stdout is redirected to stderr at startup; the
// protocol writes to a private duplicate of the original stdout, so a library log line can never
// corrupt the protocol stream.
//
// Part of Sigil (AGPL-3.0).
#include <fcntl.h>
#include <io.h>
#include <intrin.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "engine.hpp"
#include "json.hpp"

#ifndef SIGIL_BERGAMOT_VERSION
#define SIGIL_BERGAMOT_VERSION "dev"
#endif
#ifndef SIGIL_BERGAMOT_ARCH
#define SIGIL_BERGAMOT_ARCH "unknown"
#endif

namespace {

using sigil::json::Object;
using sigil::json::Value;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kProtocol = 1;
constexpr std::size_t kMaxLineBytes = 1u << 20;
constexpr std::size_t kMaxTextBytes = 64u * 1024u;
constexpr std::size_t kMaxQueued = 256;
constexpr std::size_t kMaxThreads = 8;

std::uint64_t elapsed_ms(Clock::time_point since) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - since).count());
}

void log_line(const std::string& message) {
  std::fprintf(stderr, "sigil-bergamot: %s\n", message.c_str());
  std::fflush(stderr);
}

// ---------------------------------------------------------------------------------------------
// Output channel
// ---------------------------------------------------------------------------------------------

class Output {
 public:
  explicit Output(std::FILE* f) : f_(f) {}

  void line(const std::string& s) {
    std::lock_guard<std::mutex> lock(mu_);
    std::fwrite(s.data(), 1, s.size(), f_);
    std::fputc('\n', f_);
    std::fflush(f_);
  }

  void ok(std::uint64_t id, const std::string& result) {
    line(Object().num("id", id).boolean("ok", true).raw("result", result).done());
  }

  void error(std::optional<std::uint64_t> id, const std::string& kind, const std::string& message) {
    Object o;
    if (id) o.num("id", *id); else o.null("id");
    const std::string err = Object().str("kind", kind).str("message", sigil::json::sanitize_utf8(message)).done();
    line(o.boolean("ok", false).raw("error", err).done());
  }

 private:
  std::FILE* f_;
  std::mutex mu_;
};

// ---------------------------------------------------------------------------------------------
// Line reader with a size cap
// ---------------------------------------------------------------------------------------------

class LineReader {
 public:
  /// `fd` must be in binary mode. Reads with `_read`, which returns whatever the pipe has (fread would
  /// wait until its whole buffer is full).
  explicit LineReader(int fd) : fd_(fd) {}

  enum class Result { Line, TooLong, Eof };

  Result next(std::string& out) {
    out.clear();
    bool too_long = false;
    for (;;) {
      if (pos_ == len_) {
        const int n = _read(fd_, buf_, static_cast<unsigned int>(sizeof buf_));
        len_ = n > 0 ? static_cast<std::size_t>(n) : 0;
        pos_ = 0;
        if (len_ == 0) {
          if (too_long) return Result::TooLong;
          return out.empty() ? Result::Eof : Result::Line;
        }
      }
      const char* start = buf_ + pos_;
      const void* nl = std::memchr(start, '\n', len_ - pos_);
      const std::size_t take = nl ? static_cast<std::size_t>(static_cast<const char*>(nl) - start) : len_ - pos_;
      if (!too_long) {
        if (out.size() + take > kMaxLineBytes) {
          too_long = true;
          out.clear();
        } else {
          out.append(start, take);
        }
      }
      pos_ += take;
      if (nl) {
        ++pos_;  // the newline
        if (too_long) return Result::TooLong;
        if (!out.empty() && out.back() == '\r') out.pop_back();
        return Result::Line;
      }
    }
  }

 private:
  int fd_;
  char buf_[64 * 1024];
  std::size_t pos_ = 0;
  std::size_t len_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Scheduler: loads and unloads first, then finals, then interims; at most `parallelism`
// translations run at once; queued requests can be cancelled.
// ---------------------------------------------------------------------------------------------

struct Job {
  enum class Kind { Load, Unload, Translate };
  Kind kind = Kind::Translate;
  std::uint64_t id = 0;
  std::string key;
  sigil::ModelFiles files;
  std::vector<std::string> keys;
  std::string text;
  bool final_line = true;
};

class Scheduler {
 public:
  Scheduler(sigil::Engine& engine, Output& out) : engine_(engine), out_(out) {
    worker_ = std::thread([this] { loop(); });
  }

  ~Scheduler() { shutdown(); }

  /// Queue a job; false when the queue is full.
  bool submit(Job&& job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_) return false;
      if (control_.size() + finals_.size() + interims_.size() >= kMaxQueued) return false;
      if (job.kind != Job::Kind::Translate) control_.push_back(std::move(job));
      else if (job.final_line) finals_.push_back(std::move(job));
      else interims_.push_back(std::move(job));
    }
    cv_.notify_all();
    return true;
  }

  /// Drop a queued translation (it fails with `cancelled`); false when it is not queued any more.
  bool cancel(std::uint64_t target) {
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto* q : {&finals_, &interims_}) {
        auto it = std::find_if(q->begin(), q->end(), [&](const Job& j) { return j.id == target; });
        if (it != q->end()) {
          q->erase(it);
          found = true;
          break;
        }
      }
    }
    if (found) out_.error(target, "cancelled", "cancelled");
    return found;
  }

  struct Counters {
    std::size_t queued = 0;
    std::size_t running = 0;
    std::uint64_t translations = 0;
    std::uint64_t loads = 0;
  };

  Counters counters() {
    std::lock_guard<std::mutex> lock(mu_);
    return {control_.size() + finals_.size() + interims_.size(), running_, translations_, loads_};
  }

  /// Stop taking work, drop the queue, let running translations finish (bounded), join.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_ && !worker_.joinable()) return;
      stopping_ = true;
      control_.clear();
      finals_.clear();
      interims_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, std::chrono::seconds(5), [&] { return running_ == 0; });
  }

 private:
  void loop() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      cv_.wait(lock, [&] {
        return stopping_ || !control_.empty() ||
               ((!finals_.empty() || !interims_.empty()) && running_ < std::max<std::size_t>(1, engine_.parallelism()));
      });
      if (stopping_) return;
      Job job;
      if (!control_.empty()) {
        job = std::move(control_.front());
        control_.pop_front();
        lock.unlock();
        run_control(job);
        lock.lock();
        continue;
      }
      auto& q = finals_.empty() ? interims_ : finals_;
      job = std::move(q.front());
      q.pop_front();
      ++running_;
      lock.unlock();
      run_translate(std::move(job));
      lock.lock();
    }
  }

  void finished(bool translated) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      --running_;
      if (translated) ++translations_;
    }
    cv_.notify_all();
  }

  void run_control(const Job& job) {
    const auto started = Clock::now();
    try {
      if (job.kind == Job::Kind::Load) {
        engine_.load(job.key, job.files);
        {
          std::lock_guard<std::mutex> lock(mu_);
          ++loads_;
        }
        out_.ok(job.id, Object().str("key", job.key).num("ms", elapsed_ms(started)).done());
        log_line("loaded " + job.key + " in " + std::to_string(elapsed_ms(started)) + " ms");
      } else {
        engine_.unload(job.key);
        out_.ok(job.id, Object().str("key", job.key).done());
      }
    } catch (const sigil::EngineError& e) {
      out_.error(job.id, e.kind(), e.what());
    } catch (const std::exception& e) {
      out_.error(job.id, job.kind == Job::Kind::Load ? "load_failed" : "internal", e.what());
    }
  }

  void run_translate(Job&& job) {
    const auto started = Clock::now();
    const std::uint64_t id = job.id;
    try {
      engine_.translate(job.keys, std::move(job.text), [this, id, started](sigil::Translation&& t) {
        if (t.ok) {
          out_.ok(id, Object().str("text", sigil::json::sanitize_utf8(t.text)).num("ms", elapsed_ms(started)).done());
        } else {
          out_.error(id, t.error_kind.empty() ? "translate_failed" : t.error_kind, t.error_message);
        }
        finished(t.ok);
      });
    } catch (const sigil::EngineError& e) {
      out_.error(id, e.kind(), e.what());
      finished(false);
    } catch (const std::exception& e) {
      out_.error(id, "translate_failed", e.what());
      finished(false);
    }
  }

  sigil::Engine& engine_;
  Output& out_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> control_;
  std::deque<Job> finals_;
  std::deque<Job> interims_;
  std::size_t running_ = 0;
  std::uint64_t translations_ = 0;
  std::uint64_t loads_ = 0;
  bool stopping_ = false;
  std::thread worker_;
};

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

bool valid_key(const std::string& k) {
  if (k.empty() || k.size() > 40) return false;
  return std::all_of(k.begin(), k.end(), [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'; });
}

bool string_member(const Value& msg, const char* name, std::string& out) {
  const Value* v = msg.get(name);
  if (v == nullptr || !v->is_string() || v->string.empty()) return false;
  out = v->string;
  return true;
}

struct Settings {
  std::size_t threads = 1;
  std::size_t cache_size = 0;
};

void handle(const Value& msg, std::uint64_t id, sigil::Engine& engine, Scheduler& scheduler, Output& out, const Settings& settings) {
  const std::string op = msg.get_string("op");
  if (op == "hello") {
    out.ok(id, Object()
                   .num("protocol", kProtocol)
                   .str("helper", "sigil-bergamot " SIGIL_BERGAMOT_VERSION)
                   .str("engine", engine.description())
                   .num("threads", settings.threads)
                   .str("arch", SIGIL_BERGAMOT_ARCH)
                   .done());
    return;
  }
  if (op == "stats") {
    const auto c = scheduler.counters();
    out.ok(id, Object()
                   .strings("loaded", engine.loaded())
                   .num("queued", c.queued)
                   .num("running", c.running)
                   .num("threads", settings.threads)
                   .num("translations", c.translations)
                   .num("loads", c.loads)
                   .done());
    return;
  }
  if (op == "cancel") {
    const std::uint64_t target = msg.get_u64("target", 0);
    const bool cancelled = target != 0 && scheduler.cancel(target);
    out.ok(id, Object().num("target", target).boolean("cancelled", cancelled).done());
    return;
  }
  if (op == "load" || op == "unload") {
    Job job;
    job.id = id;
    job.kind = op == "load" ? Job::Kind::Load : Job::Kind::Unload;
    job.key = msg.get_string("key");
    if (!valid_key(job.key)) return out.error(id, "invalid_request", "key must be 1-40 characters of a-z, 0-9, '_', '-'");
    if (job.kind == Job::Kind::Load) {
      if (!string_member(msg, "model", job.files.model)) return out.error(id, "invalid_request", "load needs a model path");
      if (const Value* lex = msg.get("lex"); lex != nullptr && lex->is_string()) job.files.lex = lex->string;
      const Value* vocabs = msg.get("vocabs");
      if (vocabs == nullptr || !vocabs->is_array() || vocabs->array.empty() || vocabs->array.size() > 2 ||
          !std::all_of(vocabs->array.begin(), vocabs->array.end(), [](const Value& v) { return v.is_string() && !v.string.empty(); })) {
        return out.error(id, "invalid_request", "load needs one or two vocabulary paths");
      }
      job.files.vocab_source = vocabs->array.front().string;
      job.files.vocab_target = vocabs->array.back().string;
      job.files.gemm = msg.get_string("gemm", "int8shiftAlphaAll");
      if (job.files.gemm != "int8shiftAlphaAll" && job.files.gemm != "int8shiftAll") {
        return out.error(id, "invalid_request", "gemm must be int8shiftAlphaAll or int8shiftAll");
      }
    }
    if (!scheduler.submit(std::move(job))) out.error(id, "busy", "the helper queue is full");
    return;
  }
  if (op == "translate") {
    Job job;
    job.id = id;
    const Value* keys = msg.get("keys");
    if (keys == nullptr || !keys->is_array() || keys->array.empty() || keys->array.size() > 2) {
      return out.error(id, "invalid_request", "translate needs one or two model keys");
    }
    for (const Value& k : keys->array) {
      if (!k.is_string() || !valid_key(k.string)) return out.error(id, "invalid_request", "invalid model key");
      job.keys.push_back(k.string);
    }
    for (const auto& k : job.keys) {
      if (!engine.has(k)) return out.error(id, "not_loaded", "model " + k + " is not loaded");
    }
    const Value* text = msg.get("text");
    if (text == nullptr || !text->is_string()) return out.error(id, "invalid_request", "translate needs a text");
    if (text->string.size() > kMaxTextBytes) return out.error(id, "invalid_request", "the text is too long");
    job.text = text->string;
    job.final_line = msg.get_string("priority", "final") != "interim";
    if (!scheduler.submit(std::move(job))) out.error(id, "busy", "the helper queue is full");
    return;
  }
  out.error(id, "invalid_request", "unknown op \"" + op + "\"");
}

// ---------------------------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------------------------

/// Whether this CPU can run the instruction set the helper was compiled for.
bool cpu_supported(std::string& why) {
#if defined(__AVX2__)
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 0);
  const int max_leaf = info[0];
  __cpuid(info, 1);
  const bool fma = (info[2] & (1 << 12)) != 0;
  const bool osxsave = (info[2] & (1 << 27)) != 0;
  const bool avx = (info[2] & (1 << 28)) != 0;
  bool ymm = false;
  if (osxsave) ymm = (_xgetbv(0) & 0x6) == 0x6;
  bool avx2 = false;
  if (max_leaf >= 7) {
    __cpuidex(info, 7, 0);
    avx2 = (info[1] & (1 << 5)) != 0;
  }
  if (!(avx && avx2 && fma && ymm)) {
    why = "this build needs AVX2 and FMA; use the compatibility build (build.ps1 -Arch sse2)";
    return false;
  }
#endif
  (void)why;
  return true;
}

int usage(const char* problem) {
  if (problem) std::fprintf(stderr, "sigil-bergamot: %s\n", problem);
  std::fprintf(stderr,
               "usage: sigil-bergamot.exe --jsonl [--threads N] [--cache-size N]\n"
               "       sigil-bergamot.exe --version\n"
               "JSON-lines translation helper for Sigil (protocol %llu); see docs/adr/0008-bergamot.md.\n",
               static_cast<unsigned long long>(kProtocol));
  return 2;
}

bool parse_count(const char* s, std::size_t max, std::size_t& out) {
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0' || v > max) return false;
  out = static_cast<std::size_t>(v);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool jsonl = false;
  Settings settings;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--jsonl") {
      jsonl = true;
    } else if (a == "--version") {
      std::printf("sigil-bergamot %s (%s, protocol %llu)\n", SIGIL_BERGAMOT_VERSION, SIGIL_BERGAMOT_ARCH,
                  static_cast<unsigned long long>(kProtocol));
      return 0;
    } else if (a == "--threads" && i + 1 < argc) {
      if (!parse_count(argv[++i], kMaxThreads, settings.threads) || settings.threads == 0) return usage("--threads must be 1-8");
    } else if (a == "--cache-size" && i + 1 < argc) {
      if (!parse_count(argv[++i], 1000000, settings.cache_size)) return usage("--cache-size must be 0-1000000");
    } else if (a == "--help" || a == "-h") {
      return usage(nullptr);
    } else {
      return usage(("unknown argument " + a).c_str());
    }
  }
  if (!jsonl) return usage("--jsonl is required");

  std::string why;
  if (!cpu_supported(why)) {
    log_line(why);
    return 4;
  }

  // Protocol channel: a private duplicate of stdout. Everything else that writes to stdout (library
  // logs) goes to stderr from here on.
  _setmode(_fileno(stdin), _O_BINARY);
  std::fflush(stdout);
  const int protocol_fd = _dup(_fileno(stdout));
  if (protocol_fd < 0) {
    log_line("cannot duplicate stdout");
    return 5;
  }
  _dup2(_fileno(stderr), _fileno(stdout));
  SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
  std::FILE* protocol_out = _fdopen(protocol_fd, "wb");
  if (protocol_out == nullptr) {
    log_line("cannot open the protocol channel");
    return 5;
  }
  Output out(protocol_out);

  std::unique_ptr<sigil::Engine> engine;
  try {
    engine = sigil::make_engine(sigil::EngineOptions{settings.threads, settings.cache_size});
  } catch (const std::exception& e) {
    log_line(std::string("cannot start the translation engine: ") + e.what());
    return 6;
  }
  log_line(std::string("ready: ") + engine->description() + ", " + std::to_string(settings.threads) + " thread(s), protocol " +
           std::to_string(kProtocol));

  {
    Scheduler scheduler(*engine, out);
    LineReader reader(_fileno(stdin));
    std::string line;
    for (;;) {
      const auto r = reader.next(line);
      if (r == LineReader::Result::Eof) break;
      if (r == LineReader::Result::TooLong) {
        out.error(std::nullopt, "invalid_request", "request line too long");
        continue;
      }
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      Value msg;
      std::string error;
      if (!sigil::json::parse(line, msg, error)) {
        out.error(std::nullopt, "invalid_request", "invalid JSON: " + error);
        continue;
      }
      const Value* idv = msg.get("id");
      if (!msg.is_object() || idv == nullptr || idv->type != Value::Type::Number || !idv->is_integer || idv->integer < 0) {
        out.error(std::nullopt, "invalid_request", "a request needs a non-negative integer id");
        continue;
      }
      handle(msg, static_cast<std::uint64_t>(idv->integer), *engine, scheduler, out, settings);
    }
    log_line("stdin closed, exiting");
    scheduler.shutdown();
    // Join the translation workers while the scheduler still exists: a callback of a translation that
    // outlived the bounded wait in shutdown() must never run after the scheduler is gone.
    engine.reset();
  }
  std::fflush(protocol_out);
  return 0;
}
