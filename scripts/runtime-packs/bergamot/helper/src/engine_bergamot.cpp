// The real engine: Mozilla's Bergamot (bergamot-translator in the inference/ tree of
// mozilla/translations, with Mozilla's marian fork and intgemm int8 GEMM). Models come in as memory
// bundles read here through wide-character paths, so user folders with non-ASCII names work.
//
// Upstream API used (inference/src/translator): AsyncService{Config{numWorkers, cacheSize, logger}},
// AsyncService::translate / pivot (callback with Response&&), TranslationModel(std::string config,
// MemoryBundle&&, size_t replicas), MemoryBundle{model, shortlist, vocabs}, AlignedMemory(size,
// alignment), Response::target.text, marian::setThrowExceptionOnAbort. If a re-pin changes one of these,
// this file is the only one to adapt (the protocol shell and the Rust client are independent of it).
//
// Part of Sigil (AGPL-3.0). Bergamot is MPL-2.0; Marian and intgemm are MIT (see the pack's
// THIRD-PARTY-NOTICES.md).
#include "engine.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include "common/logging.h"
#include "translator/definitions.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"
#include "translator/translation_model.h"

namespace sigil {
namespace {

namespace mb = marian::bergamot;

std::filesystem::path utf8_path(const std::string& s) {
#if defined(__cpp_char8_t)
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
  return std::filesystem::u8path(s);
#endif
}

/// Decoder settings of Firefox Translations (translations-engine.worker.js), without alignment
/// (no HTML markup is translated here) and without quality estimation.
std::string model_config(const std::string& gemm) {
  return "beam-size: 1\n"
         "normalize: 1.0\n"
         "word-penalty: 0\n"
         "max-length-break: 128\n"
         "mini-batch-words: 1024\n"
         "workspace: 128\n"
         "max-length-factor: 2.0\n"
         "skip-cost: true\n"
         "cpu-threads: 0\n"
         "quiet: true\n"
         "quiet-translation: true\n"
         "gemm-precision: " +
         gemm + "\n";
}

/// The whole file in memory with the alignment Bergamot expects (model 256, shortlist and vocabularies 64).
mb::AlignedMemory read_file(const std::string& utf8, std::size_t alignment) {
  std::ifstream in(utf8_path(utf8), std::ios::binary | std::ios::ate);
  if (!in) throw EngineError("load_failed", "cannot open " + utf8);
  const std::streamoff size = in.tellg();
  if (size <= 0) throw EngineError("load_failed", utf8 + " is empty");
  in.seekg(0, std::ios::beg);
  mb::AlignedMemory memory(static_cast<std::size_t>(size), alignment);
  in.read(memory.begin(), static_cast<std::streamsize>(size));
  if (!in) throw EngineError("load_failed", "cannot read " + utf8);
  return memory;
}

mb::AsyncService::Config service_config(const EngineOptions& options) {
  mb::AsyncService::Config config;
  config.numWorkers = options.threads == 0 ? 1 : options.threads;
  config.cacheSize = options.cache_size;
  config.logger.level = "warn";
  return config;
}

class BergamotEngine final : public Engine {
 public:
  explicit BergamotEngine(const EngineOptions& options)
      : threads_(options.threads == 0 ? 1 : options.threads), service_(service_config(options)) {}

  std::string description() const override {
#ifdef SIGIL_BERGAMOT_UPSTREAM
    return "bergamot-translator (" SIGIL_BERGAMOT_UPSTREAM ")";
#else
    return "bergamot-translator";
#endif
  }

  void load(const std::string& key, const ModelFiles& files) override {
    mb::MemoryBundle bundle;
    bundle.model = read_file(files.model, 256);
    if (!files.lex.empty()) bundle.shortlist = read_file(files.lex, 64);
    auto source = std::make_shared<mb::AlignedMemory>(read_file(files.vocab_source, 64));
    bundle.vocabs.push_back(source);
    if (files.vocab_target == files.vocab_source) {
      bundle.vocabs.push_back(source);
    } else {
      bundle.vocabs.push_back(std::make_shared<mb::AlignedMemory>(read_file(files.vocab_target, 64)));
    }
    std::shared_ptr<mb::TranslationModel> model;
    try {
      // One replica per worker thread (what AsyncService::createCompatibleModel does).
      model = std::make_shared<mb::TranslationModel>(model_config(files.gemm), std::move(bundle), threads_);
    } catch (const std::exception& e) {
      throw EngineError("load_failed", e.what());
    }
    std::lock_guard<std::mutex> lock(mu_);
    models_[key] = std::move(model);
  }

  bool unload(const std::string& key) override {
    std::lock_guard<std::mutex> lock(mu_);
    return models_.erase(key) > 0;
  }

  bool has(const std::string& key) const override {
    std::lock_guard<std::mutex> lock(mu_);
    return models_.count(key) > 0;
  }

  std::vector<std::string> loaded() const override {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> keys;
    for (const auto& [k, _] : models_) keys.push_back(k);
    return keys;
  }

  void translate(const std::vector<std::string>& keys, std::string text, TranslationCallback done) override {
    std::shared_ptr<mb::TranslationModel> first;
    std::shared_ptr<mb::TranslationModel> second;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto find = [&](const std::string& k) {
        auto it = models_.find(k);
        if (it == models_.end()) throw EngineError("not_loaded", "model " + k + " is not loaded");
        return it->second;
      };
      if (keys.empty() || keys.size() > 2) throw EngineError("invalid_request", "translate needs one or two model keys");
      first = find(keys[0]);
      if (keys.size() == 2) second = find(keys[1]);
    }
    mb::ResponseOptions options;  // plain text: no HTML, alignment or quality scores
    auto callback = [done = std::move(done)](mb::Response&& response) mutable {
      Translation t;
      t.ok = true;
      t.text = std::move(response.target.text);
      done(std::move(t));
    };
    if (second) {
      service_.pivot(first, second, std::move(text), std::move(callback), options);
    } else {
      service_.translate(first, std::move(text), std::move(callback), options);
    }
  }

  std::size_t parallelism() const override { return threads_; }

 private:
  std::size_t threads_;
  mutable std::mutex mu_;
  // Declared before the service so that the service (and its worker threads) goes first on destruction.
  std::map<std::string, std::shared_ptr<mb::TranslationModel>> models_;
  mb::AsyncService service_;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const EngineOptions& options) {
  // Marian aborts the process on configuration errors by default; exceptions let a bad model fail its
  // `load` request instead of taking the helper down.
  marian::setThrowExceptionOnAbort(true);
  return std::make_unique<BergamotEngine>(options);
}

}  // namespace sigil
