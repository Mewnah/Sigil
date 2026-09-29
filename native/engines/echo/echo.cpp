// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_echo.dll - the sigil_engine ABI v1 test engine.
//
// It proves the whole native-runtime path (build presets, packaging, hash-verified lazy load, ABI
// check, variant selection, calls, streams, cancellation, logging) without any model or GPU:
//   create   {"gain": 1.0}                        gain in [-16, 16]; also {"fail": "load" | "backend"}
//                                                 to make create fail with SIGIL_E_LOAD_FAILED /
//                                                 SIGIL_E_BACKEND_UNAVAILABLE (host fallback tests)
//   query    {"op":"info"}                        name, version, variant, backend, gain
//            {"op":"config"}                      {"gain": g}
//            {"op":"sleep","ms":N}                sleeps up to N ms, cancellable
//   process  audio, {"work_ms": N}                statistics: frames, rms, peak, ...
//   render   audio, {"gain": g}                   the input times the gain
//            NULL,  {"freq","ms","sample_rate","channels","amplitude"}   a sine tone
//   stream   push -> {"partial": "...", "frames": n}; finish -> {"text": "...", "frames": n, "rms": r}
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "sigil_engine_impl.hpp"

#ifndef SIGIL_ENGINE_VERSION
#error "SIGIL_ENGINE_VERSION is set by sigil_add_engine()"
#endif

namespace {

constexpr double kMaxGain = 16.0;
constexpr double kPi = 3.14159265358979323846;

double checked_gain(double g) {
    if (!std::isfinite(g) || std::fabs(g) > kMaxGain) {
        sigil::fail(SIGIL_E_INVALID_ARGUMENT, "gain must be a finite number between -16 and 16");
    }
    return g;
}

struct Stats {
    double sum_sq = 0.0;
    double peak = 0.0;
    uint64_t samples = 0;

    void add(const sigil::Audio& a) {
        const size_t n = a.sample_count();
        for (size_t i = 0; i < n; ++i) {
            const double s = a.samples[i];
            sum_sq += s * s;
            peak = std::max(peak, std::fabs(s));
        }
        samples += n;
    }
    double rms() const { return samples ? std::sqrt(sum_sq / static_cast<double>(samples)) : 0.0; }
};

class EchoStream final : public sigil::Stream {
public:
    std::optional<std::string> push(const sigil::Audio& audio) override {
        frames_ += audio.frames;
        stats_.add(audio);
        sigil::json::Writer w;
        w.begin_object()
            .field("partial", std::to_string(frames_) + " frames")
            .field("frames", frames_)
            .end_object();
        return w.take();
    }

    std::string finish() override {
        sigil::json::Writer w;
        w.begin_object()
            .field("text", "echo: " + std::to_string(frames_) + " frames")
            .field("frames", frames_)
            .field("rms", stats_.rms())
            .end_object();
        frames_ = 0;
        stats_ = Stats{};
        return w.take();
    }

private:
    uint64_t frames_ = 0;
    Stats stats_;
};

class EchoEngine final : public sigil::Engine {
public:
    explicit EchoEngine(double gain) : gain_(gain) {}

    std::string query(const sigil::json::Value& request) override {
        const std::string op = request.string_or("op", "info");
        sigil::json::Writer w;
        if (op == "info") {
            const sigil::EngineDescriptor& d = sigil::engine_descriptor();
            w.begin_object()
                .field("name", d.name)
                .field("version", d.version)
                .field("variant", d.variant)
                .field("backend", backend())
                .field("gain", gain_)
                .end_object();
        } else if (op == "config") {
            w.begin_object().field("gain", gain_).end_object();
        } else if (op == "sleep") {
            const double ms = std::clamp(request.number_or("ms", 0.0), 0.0, 60000.0);
            sigil::sleep_cancellable(cancel_token(), ms);
            w.begin_object().field("slept_ms", ms).end_object();
        } else {
            sigil::fail(SIGIL_E_INVALID_ARGUMENT, "unknown query op \"" + op + "\"");
        }
        return w.take();
    }

    std::string process(const sigil::Audio& audio, const sigil::json::Value& options) override {
        sigil::sleep_cancellable(cancel_token(), std::clamp(options.number_or("work_ms", 0.0), 0.0, 60000.0));
        Stats stats;
        stats.add(audio);
        sigil::json::Writer w;
        w.begin_object()
            .field("frames", audio.frames)
            .field("sample_rate", audio.sample_rate)
            .field("channels", audio.channels)
            .field("duration_ms", audio.seconds() * 1000.0)
            .field("rms", stats.rms())
            .field("peak", stats.peak)
            .end_object();
        return w.take();
    }

    sigil::AudioOut render(const sigil::Audio* input, const sigil::json::Value& options) override {
        sigil::AudioOut out;
        if (input) {
            const float gain = static_cast<float>(checked_gain(options.number_or("gain", gain_)));
            const size_t n = input->sample_count();
            out.samples.resize(n);
            for (size_t i = 0; i < n; ++i) out.samples[i] = input->samples[i] * gain;
            out.sample_rate = input->sample_rate;
            out.channels = input->channels;
            return out;
        }
        // Generator: a sine tone.
        const double freq = options.number_or("freq", 440.0);
        const double ms = options.number_or("ms", 100.0);
        const double rate = options.number_or("sample_rate", 16000.0);
        const double channels = options.number_or("channels", 1.0);
        const double amplitude = options.number_or("amplitude", 0.5);
        if (!(freq > 0.0 && freq < 100000.0) || !(ms >= 0.0 && ms <= 600000.0) || !(rate >= 8000.0 && rate <= 384000.0) ||
            !(channels >= 1.0 && channels <= 8.0) || !(std::fabs(amplitude) <= 1.0)) {
            sigil::fail(SIGIL_E_INVALID_ARGUMENT, "invalid tone options");
        }
        const auto sr = static_cast<uint32_t>(rate);
        const auto ch = static_cast<uint32_t>(channels);
        const auto frames = static_cast<size_t>(std::llround(ms * sr / 1000.0));
        out.samples.resize(frames * ch);
        for (size_t f = 0; f < frames; ++f) {
            if ((f & 0xFFF) == 0) cancel_token().check();
            const auto s = static_cast<float>(amplitude * std::sin(2.0 * kPi * freq * static_cast<double>(f) / sr));
            for (uint32_t c = 0; c < ch; ++c) out.samples[f * ch + c] = s;
        }
        out.sample_rate = sr;
        out.channels = ch;
        return out;
    }

    std::unique_ptr<sigil::Stream> open_stream(const sigil::json::Value&) override { return std::make_unique<EchoStream>(); }

private:
    double gain_;
};

}  // namespace

const sigil::EngineDescriptor& sigil::engine_descriptor() {
    static const EngineDescriptor d{
        "echo",
        SIGIL_ENGINE_VERSION,
        SIGIL_ENGINE_VARIANT,
        "test",
        SIGIL_ENGINE_BUILD_INFO,
        SIGIL_CAP_PROCESS | SIGIL_CAP_STREAM | SIGIL_CAP_RENDER | SIGIL_CAP_CANCEL,
    };
    return d;
}

std::unique_ptr<sigil::Engine> sigil::create_engine(const json::Value& config) {
    const std::string failure = config.string_or("fail", "");
    if (failure == "load") fail(SIGIL_E_LOAD_FAILED, "echo: load failure requested by the config");
    if (failure == "backend") fail(SIGIL_E_BACKEND_UNAVAILABLE, "echo: backend unavailable (requested by the config)");
    if (!failure.empty()) fail(SIGIL_E_INVALID_ARGUMENT, "echo: unknown \"fail\" value \"" + failure + "\"");
    const double gain = checked_gain(config.number_or("gain", 1.0));
    sigil::log(SIGIL_LOG_DEBUG, "echo: created (gain " + std::to_string(gain) + ")");
    return std::make_unique<EchoEngine>(gain);
}
