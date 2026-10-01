// Minimal JSON for the sigil-bergamot protocol: parse one request line, build reply lines.
// Strict RFC 8259 parsing (no comments, no trailing commas), depth-limited, UTF-8 in and out.
// Part of Sigil (AGPL-3.0); see scripts/runtime-packs/bergamot/helper/README.md.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sigil::json {

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  /// Set when the number was written as an integer that fits (`is_integer`).
  bool is_integer = false;
  std::int64_t integer = 0;
  std::string string;
  std::vector<Value> array;
  std::vector<std::pair<std::string, Value>> object;

  bool is_null() const { return type == Type::Null; }
  bool is_string() const { return type == Type::String; }
  bool is_array() const { return type == Type::Array; }
  bool is_object() const { return type == Type::Object; }

  /// Member of an object (nullptr when absent or not an object). The last duplicate wins.
  const Value* get(std::string_view key) const;
  /// Non-negative integer member, or `fallback`.
  std::uint64_t get_u64(std::string_view key, std::uint64_t fallback) const;
  /// String member, or `fallback`.
  std::string get_string(std::string_view key, std::string_view fallback = {}) const;
};

/// Parse `text` (one JSON document, surrounding whitespace allowed). False with `error` set on failure.
bool parse(std::string_view text, Value& out, std::string& error);

/// Append `s` as a JSON string literal. Invalid UTF-8 bytes become U+FFFD, so the output is always valid.
void append_string(std::string& out, std::string_view s);

/// Replace invalid UTF-8 sequences with U+FFFD.
std::string sanitize_utf8(std::string_view s);

/// Small builder for reply objects: `Object().str("k", v).num("n", 1).done()`.
class Object {
 public:
  Object& str(std::string_view key, std::string_view value);
  Object& num(std::string_view key, std::uint64_t value);
  Object& boolean(std::string_view key, bool value);
  Object& null(std::string_view key);
  /// `value` must already be valid JSON.
  Object& raw(std::string_view key, std::string_view value);
  Object& strings(std::string_view key, const std::vector<std::string>& values);
  std::string done() const { return body_ + "}"; }

 private:
  void key(std::string_view k);
  std::string body_ = "{";
  bool first_ = true;
};

}  // namespace sigil::json
