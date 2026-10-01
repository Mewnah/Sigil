// Translation engine behind the sigil-bergamot protocol. The real one wraps Mozilla's Bergamot
// (engine_bergamot.cpp); the fake one (engine_fake.cpp, -DSIGIL_BERGAMOT_FAKE=ON) lets the protocol
// shell be built and tested without the upstream sources. Part of Sigil (AGPL-3.0).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sigil {

/// A failure with a protocol error kind (`load_failed`, `not_loaded`, `translate_failed`, ...).
class EngineError : public std::runtime_error {
 public:
  EngineError(std::string kind, const std::string& message) : std::runtime_error(message), kind_(std::move(kind)) {}
  const std::string& kind() const { return kind_; }

 private:
  std::string kind_;
};

struct EngineOptions {
  /// Translation worker threads (each keeps its own model replica workspace).
  std::size_t threads = 1;
  /// Translation cache entries (sentences); 0 disables the cache.
  std::size_t cache_size = 0;
};

/// Absolute UTF-8 paths of one direction's files.
struct ModelFiles {
  std::string model;
  /// Empty: no shortlist (slower, but works).
  std::string lex;
  /// Source and target vocabulary (the same path twice for a shared vocabulary).
  std::string vocab_source;
  std::string vocab_target;
  /// `int8shiftAlphaAll` (`*.intgemm.alphas.bin`) or `int8shiftAll` (`*.intgemm8.bin`).
  std::string gemm;
};

/// Result of one translation: `ok` with the text, or an error kind and message.
struct Translation {
  bool ok = false;
  std::string text;
  std::string error_kind;
  std::string error_message;
};

using TranslationCallback = std::function<void(Translation&&)>;

class Engine {
 public:
  virtual ~Engine() = default;

  /// `bergamot-translator (mozilla/translations <commit>)` or `fake`.
  virtual std::string description() const = 0;

  /// Load (or replace) the model `key`. Throws EngineError("load_failed", ...) on failure.
  virtual void load(const std::string& key, const ModelFiles& files) = 0;

  /// Forget `key`; translations already running keep their model alive. False when unknown.
  virtual bool unload(const std::string& key) = 0;

  virtual bool has(const std::string& key) const = 0;
  virtual std::vector<std::string> loaded() const = 0;

  /// Translate `text` with `keys` (one model, or two for a pivot through the first model's target
  /// language). Calls `done` exactly once, possibly on another thread. Throws EngineError
  /// ("not_loaded") before starting when a key is unknown.
  virtual void translate(const std::vector<std::string>& keys, std::string text, TranslationCallback done) = 0;

  /// How many translations may run at the same time.
  virtual std::size_t parallelism() const = 0;
};

std::unique_ptr<Engine> make_engine(const EngineOptions& options);

}  // namespace sigil
