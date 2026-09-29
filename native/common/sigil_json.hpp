// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_json.hpp - minimal JSON value, parser and writer for sigil_engine DLLs.
//
// Engines receive their configuration and options as JSON (sigil_engine.h) and answer with JSON.
// This is a small, dependency-free implementation for flat configuration objects: it parses any
// RFC 8259 document (depth-limited, \uXXXX escapes including surrogate pairs, locale-independent
// numbers) and writes compact JSON with correct escaping. Engines with heavier needs may bring
// their own library; the ABI only sees strings.
#ifndef SIGIL_JSON_HPP
#define SIGIL_JSON_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sigil::json {

/** A parsed JSON value. Objects keep their key order; duplicate keys: the last one wins in find(). */
class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;
    static Value null() { return Value(); }
    static Value boolean(bool b);
    static Value number(double n);
    static Value string(std::string s);
    static Value array(std::vector<Value> items);
    static Value object(std::vector<std::pair<std::string, Value>> members);

    /** Parse `text` (NULL-free). Throws std::invalid_argument with a position on malformed input. */
    static Value parse(std::string_view text);

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::Null; }
    bool is_bool() const noexcept { return type_ == Type::Bool; }
    bool is_number() const noexcept { return type_ == Type::Number; }
    bool is_string() const noexcept { return type_ == Type::String; }
    bool is_array() const noexcept { return type_ == Type::Array; }
    bool is_object() const noexcept { return type_ == Type::Object; }

    bool as_bool() const noexcept { return bool_; }
    double as_number() const noexcept { return number_; }
    const std::string& as_string() const noexcept { return string_; }
    const std::vector<Value>& items() const noexcept { return items_; }
    const std::vector<std::pair<std::string, Value>>& members() const noexcept { return members_; }

    /** Member of an object (last occurrence), or nullptr (also for non-objects). */
    const Value* find(std::string_view key) const noexcept;

    /** Typed member lookups with a default for missing members; a member of the wrong type throws
     *  std::invalid_argument naming the key. */
    double number_or(std::string_view key, double fallback) const;
    std::string string_or(std::string_view key, std::string fallback) const;
    bool bool_or(std::string_view key, bool fallback) const;

    /** Compact JSON text of this value. */
    std::string dump() const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> items_;
    std::vector<std::pair<std::string, Value>> members_;
};

/** Append `s` to `out` as a JSON string literal (quotes included; control characters escaped). */
void append_escaped(std::string& out, std::string_view s);

/** Append a number: shortest round-trip form; NaN and infinities become `null`. */
void append_number(std::string& out, double n);

/**
 * Streaming writer for compact JSON objects:
 *   Writer w; w.begin_object().field("name", "echo").field("frames", 480).end_object();
 *   std::string text = w.take();
 * Commas are inserted automatically. Misuse (a value without a key inside an object) is not
 * diagnosed; keep writers simple and linear.
 */
class Writer {
public:
    Writer& begin_object();
    Writer& end_object();
    Writer& begin_array();
    Writer& end_array();
    Writer& key(std::string_view k);
    Writer& value(std::string_view s);
    Writer& value(const char* s) { return value(std::string_view(s ? s : "")); }
    Writer& value(const std::string& s) { return value(std::string_view(s)); }
    Writer& value(double n);
    Writer& value(int64_t n);
    Writer& value(uint64_t n);
    Writer& value(int n) { return value(static_cast<int64_t>(n)); }
    Writer& value(unsigned n) { return value(static_cast<uint64_t>(n)); }
    Writer& value(bool b);
    Writer& null();
    /** Insert pre-serialized JSON as a value. */
    Writer& raw(std::string_view json);

    template <class T>
    Writer& field(std::string_view k, const T& v) {
        key(k);
        return value(v);
    }

    const std::string& str() const noexcept { return out_; }
    std::string take() { return std::move(out_); }

private:
    void before_value();
    std::string out_;
    std::vector<bool> first_;  // one entry per open container: no element written yet
    bool after_key_ = false;
};

}  // namespace sigil::json

#endif  // SIGIL_JSON_HPP
