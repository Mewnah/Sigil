// Minimal JSON for the sigil-bergamot protocol. Part of Sigil (AGPL-3.0).
#include "json.hpp"

#include <charconv>
#include <cstdlib>

namespace sigil::json {
namespace {

constexpr int kMaxDepth = 32;

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

/// Length of the valid UTF-8 sequence at `s[i]`, or 0 when it is invalid.
std::size_t utf8_sequence(std::string_view s, std::size_t i) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) return 1;
  std::size_t len = 0;
  std::uint32_t cp = 0;
  if ((b0 & 0xE0) == 0xC0) {
    len = 2;
    cp = b0 & 0x1F;
  } else if ((b0 & 0xF0) == 0xE0) {
    len = 3;
    cp = b0 & 0x0F;
  } else if ((b0 & 0xF8) == 0xF0) {
    len = 4;
    cp = b0 & 0x07;
  } else {
    return 0;
  }
  if (i + len > s.size()) return 0;
  for (std::size_t k = 1; k < len; ++k) {
    const auto b = static_cast<unsigned char>(s[i + k]);
    if ((b & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (b & 0x3F);
  }
  // Overlong forms, surrogates and values beyond U+10FFFF are invalid.
  if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)) return 0;
  if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
  if (cp > 0x10FFFF) return 0;
  return len;
}

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  bool document(Value& out, std::string& error) {
    skip_ws();
    if (!value(out, 0)) {
      error = error_.empty() ? "invalid JSON" : error_;
      return false;
    }
    skip_ws();
    if (pos_ != s_.size()) {
      error = "unexpected data after the JSON value at byte " + std::to_string(pos_);
      return false;
    }
    return true;
  }

 private:
  bool fail(const char* what) {
    if (error_.empty()) error_ = std::string(what) + " at byte " + std::to_string(pos_);
    return false;
  }

  void skip_ws() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
  }

  bool literal(std::string_view word) {
    if (s_.substr(pos_, word.size()) != word) return fail("invalid literal");
    pos_ += word.size();
    return true;
  }

  bool value(Value& out, int depth) {
    if (depth > kMaxDepth) return fail("nesting too deep");
    if (pos_ >= s_.size()) return fail("unexpected end");
    switch (s_[pos_]) {
      case '{':
        return object(out, depth);
      case '[':
        return array(out, depth);
      case '"':
        out.type = Value::Type::String;
        return string(out.string);
      case 't':
        out.type = Value::Type::Bool;
        out.boolean = true;
        return literal("true");
      case 'f':
        out.type = Value::Type::Bool;
        out.boolean = false;
        return literal("false");
      case 'n':
        out.type = Value::Type::Null;
        return literal("null");
      default:
        return number(out);
    }
  }

  bool object(Value& out, int depth) {
    out.type = Value::Type::Object;
    ++pos_;  // {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == '}') {
      ++pos_;
      return true;
    }
    for (;;) {
      skip_ws();
      if (pos_ >= s_.size() || s_[pos_] != '"') return fail("expected a member name");
      std::string key;
      if (!string(key)) return false;
      skip_ws();
      if (pos_ >= s_.size() || s_[pos_] != ':') return fail("expected ':'");
      ++pos_;
      skip_ws();
      Value v;
      if (!value(v, depth + 1)) return false;
      out.object.emplace_back(std::move(key), std::move(v));
      skip_ws();
      if (pos_ < s_.size() && s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < s_.size() && s_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }

  bool array(Value& out, int depth) {
    out.type = Value::Type::Array;
    ++pos_;  // [
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == ']') {
      ++pos_;
      return true;
    }
    for (;;) {
      skip_ws();
      Value v;
      if (!value(v, depth + 1)) return false;
      out.array.push_back(std::move(v));
      skip_ws();
      if (pos_ < s_.size() && s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < s_.size() && s_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }

  bool hex4(std::uint32_t& cp) {
    if (pos_ + 4 > s_.size()) return fail("truncated \\u escape");
    cp = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      cp <<= 4;
      if (c >= '0' && c <= '9') cp |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') cp |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') cp |= static_cast<std::uint32_t>(c - 'A' + 10);
      else return fail("invalid \\u escape");
    }
    return true;
  }

  bool string(std::string& out) {
    ++pos_;  // opening quote
    for (;;) {
      if (pos_ >= s_.size()) return fail("unterminated string");
      const char c = s_[pos_];
      if (c == '"') {
        ++pos_;
        return true;
      }
      if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
      if (c != '\\') {
        const std::size_t len = utf8_sequence(s_, pos_);
        if (len == 0) {
          append_utf8(out, 0xFFFD);
          ++pos_;
        } else {
          out.append(s_.substr(pos_, len));
          pos_ += len;
        }
        continue;
      }
      ++pos_;
      if (pos_ >= s_.size()) return fail("unterminated escape");
      const char e = s_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          std::uint32_t cp = 0;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            // A high surrogate must be followed by \uDC00-\uDFFF; a lone one becomes U+FFFD.
            if (pos_ + 6 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              const std::size_t save = pos_;
              pos_ += 2;
              std::uint32_t lo = 0;
              if (!hex4(lo)) return false;
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                pos_ = save;
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          append_utf8(out, cp);
          break;
        }
        default:
          return fail("invalid escape");
      }
    }
  }

  bool number(Value& out) {
    const std::size_t start = pos_;
    if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
    if (pos_ >= s_.size() || s_[pos_] < '0' || s_[pos_] > '9') return fail("invalid value");
    if (s_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    bool integral = true;
    if (pos_ < s_.size() && s_[pos_] == '.') {
      integral = false;
      ++pos_;
      if (pos_ >= s_.size() || s_[pos_] < '0' || s_[pos_] > '9') return fail("invalid number");
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      integral = false;
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
      if (pos_ >= s_.size() || s_[pos_] < '0' || s_[pos_] > '9') return fail("invalid number");
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    const std::string_view token = s_.substr(start, pos_ - start);
    out.type = Value::Type::Number;
    out.number = std::strtod(std::string(token).c_str(), nullptr);
    if (integral) {
      std::int64_t v = 0;
      const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), v);
      out.is_integer = ec == std::errc() && ptr == token.data() + token.size();
      out.integer = out.is_integer ? v : 0;
    }
    return true;
  }

  std::string_view s_;
  std::size_t pos_ = 0;
  std::string error_;
};

}  // namespace

const Value* Value::get(std::string_view key) const {
  if (type != Type::Object) return nullptr;
  const Value* found = nullptr;
  for (const auto& [k, v] : object) {
    if (k == key) found = &v;
  }
  return found;
}

std::uint64_t Value::get_u64(std::string_view key, std::uint64_t fallback) const {
  const Value* v = get(key);
  if (v == nullptr || v->type != Type::Number || !v->is_integer || v->integer < 0) return fallback;
  return static_cast<std::uint64_t>(v->integer);
}

std::string Value::get_string(std::string_view key, std::string_view fallback) const {
  const Value* v = get(key);
  if (v == nullptr || v->type != Type::String) return std::string(fallback);
  return v->string;
}

bool parse(std::string_view text, Value& out, std::string& error) {
  out = Value{};
  Parser p(text);
  return p.document(out, error);
}

std::string sanitize_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const std::size_t len = utf8_sequence(s, i);
    if (len == 0) {
      append_utf8(out, 0xFFFD);
      ++i;
    } else {
      out.append(s.substr(i, len));
      i += len;
    }
  }
  return out;
}

void append_string(std::string& out, std::string_view s) {
  static const char* kHex = "0123456789abcdef";
  out.push_back('"');
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
          if (c < 0x20 || c == 0x7F) {
            out += "\\u00";
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 0xF]);
          } else {
            out.push_back(static_cast<char>(c));
          }
      }
      ++i;
      continue;
    }
    const std::size_t len = utf8_sequence(s, i);
    if (len == 0) {
      append_utf8(out, 0xFFFD);
      ++i;
    } else {
      // U+2028 / U+2029 are valid JSON but break some line-oriented readers: escape them.
      if (len == 3 && c == 0xE2 && static_cast<unsigned char>(s[i + 1]) == 0x80 &&
          (static_cast<unsigned char>(s[i + 2]) == 0xA8 || static_cast<unsigned char>(s[i + 2]) == 0xA9)) {
        out += static_cast<unsigned char>(s[i + 2]) == 0xA8 ? "\\u2028" : "\\u2029";
      } else {
        out.append(s.substr(i, len));
      }
      i += len;
    }
  }
  out.push_back('"');
}

void Object::key(std::string_view k) {
  if (!first_) body_.push_back(',');
  first_ = false;
  append_string(body_, k);
  body_.push_back(':');
}

Object& Object::str(std::string_view k, std::string_view value) {
  key(k);
  append_string(body_, value);
  return *this;
}

Object& Object::num(std::string_view k, std::uint64_t value) {
  key(k);
  body_ += std::to_string(value);
  return *this;
}

Object& Object::boolean(std::string_view k, bool value) {
  key(k);
  body_ += value ? "true" : "false";
  return *this;
}

Object& Object::null(std::string_view k) {
  key(k);
  body_ += "null";
  return *this;
}

Object& Object::raw(std::string_view k, std::string_view value) {
  key(k);
  body_.append(value);
  return *this;
}

Object& Object::strings(std::string_view k, const std::vector<std::string>& values) {
  key(k);
  body_.push_back('[');
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) body_.push_back(',');
    append_string(body_, values[i]);
  }
  body_.push_back(']');
  return *this;
}

}  // namespace sigil::json
