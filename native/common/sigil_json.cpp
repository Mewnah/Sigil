// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_json.cpp - see sigil_json.hpp.
#include "sigil_json.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <system_error>

namespace sigil::json {

namespace {

constexpr int kMaxDepth = 64;

class Parser {
public:
    explicit Parser(std::string_view text) : s_(text) {}

    Value parse_document() {
        skip_ws();
        Value v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) fail("unexpected trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::invalid_argument(std::string("invalid JSON at offset ") + std::to_string(pos_) + ": " + what);
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }

    bool consume(char c) {
        if (pos_ < s_.size() && s_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect_word(std::string_view word) {
        if (s_.substr(pos_, word.size()) != word) fail("unknown literal");
        pos_ += word.size();
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting too deep");
        if (pos_ >= s_.size()) fail("unexpected end of input");
        switch (s_[pos_]) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': return Value::string(parse_string());
            case 't': expect_word("true"); return Value::boolean(true);
            case 'f': expect_word("false"); return Value::boolean(false);
            case 'n': expect_word("null"); return Value::null();
            default: return Value::number(parse_number());
        }
    }

    Value parse_object(int depth) {
        ++pos_;  // '{'
        std::vector<std::pair<std::string, Value>> members;
        skip_ws();
        if (consume('}')) return Value::object(std::move(members));
        for (;;) {
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a string key");
            std::string key = parse_string();
            skip_ws();
            if (!consume(':')) fail("expected ':'");
            skip_ws();
            Value v = parse_value(depth + 1);
            members.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (consume(',')) continue;
            if (consume('}')) break;
            fail("expected ',' or '}'");
        }
        return Value::object(std::move(members));
    }

    Value parse_array(int depth) {
        ++pos_;  // '['
        std::vector<Value> items;
        skip_ws();
        if (consume(']')) return Value::array(std::move(items));
        for (;;) {
            skip_ws();
            items.push_back(parse_value(depth + 1));
            skip_ws();
            if (consume(',')) continue;
            if (consume(']')) break;
            fail("expected ',' or ']'");
        }
        return Value::array(std::move(items));
    }

    unsigned hex4() {
        if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') break;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 2 > s_.size() || s_[pos_] != '\\' || s_[pos_ + 1] != 'u') fail("unpaired surrogate");
                        pos_ += 2;
                        const unsigned lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        return out;
    }

    double parse_number() {
        const size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
        if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("expected a value");
        if (s_[pos_] == '0') {
            ++pos_;
        } else {
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("digit expected after '.'");
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("digit expected in exponent");
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        double v = 0.0;
        const char* first = s_.data() + start;
        const char* last = s_.data() + pos_;
        const auto res = std::from_chars(first, last, v);
        if (res.ec == std::errc::result_out_of_range) {
            // Beyond double: a negative exponent underflows to (signed) zero, anything else
            // overflows to (signed) infinity; engines reject non-finite options themselves.
            const std::string_view tok(first, static_cast<size_t>(last - first));
            const size_t e = tok.find_first_of("eE");
            const bool underflow = e != std::string_view::npos && e + 1 < tok.size() && tok[e + 1] == '-';
            const bool negative = tok.front() == '-';
            v = underflow ? (negative ? -0.0 : 0.0) : (negative ? -HUGE_VAL : HUGE_VAL);
        } else if (res.ec != std::errc() || res.ptr != last) {
            fail("invalid number");
        }
        return v;
    }

    std::string_view s_;
    size_t pos_ = 0;
};

}  // namespace

Value Value::boolean(bool b) {
    Value v;
    v.type_ = Type::Bool;
    v.bool_ = b;
    return v;
}

Value Value::number(double n) {
    Value v;
    v.type_ = Type::Number;
    v.number_ = n;
    return v;
}

Value Value::string(std::string s) {
    Value v;
    v.type_ = Type::String;
    v.string_ = std::move(s);
    return v;
}

Value Value::array(std::vector<Value> items) {
    Value v;
    v.type_ = Type::Array;
    v.items_ = std::move(items);
    return v;
}

Value Value::object(std::vector<std::pair<std::string, Value>> members) {
    Value v;
    v.type_ = Type::Object;
    v.members_ = std::move(members);
    return v;
}

Value Value::parse(std::string_view text) { return Parser(text).parse_document(); }

const Value* Value::find(std::string_view key) const noexcept {
    if (type_ != Type::Object) return nullptr;
    for (auto it = members_.rbegin(); it != members_.rend(); ++it) {
        if (it->first == key) return &it->second;
    }
    return nullptr;
}

static std::invalid_argument wrong_type(std::string_view key, const char* expected) {
    return std::invalid_argument("\"" + std::string(key) + "\" must be " + expected);
}

double Value::number_or(std::string_view key, double fallback) const {
    const Value* v = find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_number()) throw wrong_type(key, "a number");
    return v->as_number();
}

std::string Value::string_or(std::string_view key, std::string fallback) const {
    const Value* v = find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_string()) throw wrong_type(key, "a string");
    return v->as_string();
}

bool Value::bool_or(std::string_view key, bool fallback) const {
    const Value* v = find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_bool()) throw wrong_type(key, "true or false");
    return v->as_bool();
}

std::string Value::dump() const {
    std::string out;
    switch (type_) {
        case Type::Null: out = "null"; break;
        case Type::Bool: out = bool_ ? "true" : "false"; break;
        case Type::Number: append_number(out, number_); break;
        case Type::String: append_escaped(out, string_); break;
        case Type::Array: {
            out += '[';
            for (size_t i = 0; i < items_.size(); ++i) {
                if (i) out += ',';
                out += items_[i].dump();
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            for (size_t i = 0; i < members_.size(); ++i) {
                if (i) out += ',';
                append_escaped(out, members_[i].first);
                out += ':';
                out += members_[i].second.dump();
            }
            out += '}';
            break;
        }
    }
    return out;
}

void append_escaped(std::string& out, std::string_view s) {
    static const char* hex = "0123456789abcdef";
    out += '"';
    for (const char c : s) {
        const auto u = static_cast<unsigned char>(c);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (u < 0x20) {
                    out += "\\u00";
                    out += hex[u >> 4];
                    out += hex[u & 0xF];
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

void append_number(std::string& out, double n) {
    if (!std::isfinite(n)) {
        out += "null";
        return;
    }
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof buf, n);
    if (res.ec != std::errc()) {
        out += "null";
        return;
    }
    out.append(buf, res.ptr);
}

void Writer::before_value() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (!first_.empty()) {
        if (!first_.back()) out_ += ',';
        first_.back() = false;
    }
}

Writer& Writer::begin_object() {
    before_value();
    out_ += '{';
    first_.push_back(true);
    return *this;
}

Writer& Writer::end_object() {
    out_ += '}';
    if (!first_.empty()) first_.pop_back();
    return *this;
}

Writer& Writer::begin_array() {
    before_value();
    out_ += '[';
    first_.push_back(true);
    return *this;
}

Writer& Writer::end_array() {
    out_ += ']';
    if (!first_.empty()) first_.pop_back();
    return *this;
}

Writer& Writer::key(std::string_view k) {
    if (!first_.empty()) {
        if (!first_.back()) out_ += ',';
        first_.back() = false;
    }
    append_escaped(out_, k);
    out_ += ':';
    after_key_ = true;
    return *this;
}

Writer& Writer::value(std::string_view s) {
    before_value();
    append_escaped(out_, s);
    return *this;
}

Writer& Writer::value(double n) {
    before_value();
    append_number(out_, n);
    return *this;
}

Writer& Writer::value(int64_t n) {
    before_value();
    out_ += std::to_string(n);
    return *this;
}

Writer& Writer::value(uint64_t n) {
    before_value();
    out_ += std::to_string(n);
    return *this;
}

Writer& Writer::value(bool b) {
    before_value();
    out_ += b ? "true" : "false";
    return *this;
}

Writer& Writer::null() {
    before_value();
    out_ += "null";
    return *this;
}

Writer& Writer::raw(std::string_view json) {
    before_value();
    out_.append(json.data(), json.size());
    return *this;
}

}  // namespace sigil::json
