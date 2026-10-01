// Fake engine for protocol tests (-DSIGIL_BERGAMOT_FAKE=ON): no upstream sources needed.
// A translation is "<key1+key2>text"; "__crash__" in the text ends the process (exit 3) and "__slow__"
// sleeps 1.5 s first, like the Rust test fake (mt_local/bergamot/tests.rs). Loading checks that the
// files exist. Part of Sigil (AGPL-3.0).
#include "engine.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>

namespace sigil {
namespace {

std::filesystem::path utf8_path(const std::string& s) {
#if defined(__cpp_char8_t)
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
  return std::filesystem::u8path(s);
#endif
}

class FakeEngine final : public Engine {
 public:
  explicit FakeEngine(const EngineOptions& options) : options_(options) {}

  std::string description() const override { return "fake"; }

  void load(const std::string& key, const ModelFiles& files) override {
    for (const std::string* p : {&files.model, &files.vocab_source, &files.vocab_target}) {
      std::error_code ec;
      if (!std::filesystem::is_regular_file(utf8_path(*p), ec)) throw EngineError("load_failed", "cannot open " + *p);
    }
    if (!files.lex.empty()) {
      std::error_code ec;
      if (!std::filesystem::is_regular_file(utf8_path(files.lex), ec)) throw EngineError("load_failed", "cannot open " + files.lex);
    }
    std::lock_guard<std::mutex> lock(mu_);
    keys_.insert(key);
  }

  bool unload(const std::string& key) override {
    std::lock_guard<std::mutex> lock(mu_);
    return keys_.erase(key) > 0;
  }

  bool has(const std::string& key) const override {
    std::lock_guard<std::mutex> lock(mu_);
    return keys_.count(key) > 0;
  }

  std::vector<std::string> loaded() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return {keys_.begin(), keys_.end()};
  }

  void translate(const std::vector<std::string>& keys, std::string text, TranslationCallback done) override {
    std::string joined;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (const auto& k : keys) {
        if (!keys_.count(k)) throw EngineError("not_loaded", "model " + k + " is not loaded");
        if (!joined.empty()) joined += '+';
        joined += k;
      }
    }
    if (text.find("__crash__") != std::string::npos) std::exit(3);
    if (text.find("__slow__") != std::string::npos) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    Translation t;
    t.ok = true;
    t.text = "<" + joined + ">" + text;
    done(std::move(t));
  }

  std::size_t parallelism() const override { return 1; }

 private:
  EngineOptions options_;
  mutable std::mutex mu_;
  std::set<std::string> keys_;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const EngineOptions& options) { return std::make_unique<FakeEngine>(options); }

}  // namespace sigil
