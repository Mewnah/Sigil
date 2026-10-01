// Unit tests of the helper's JSON code (built and run by `build.ps1 -Fake`). Part of Sigil (AGPL-3.0).
#include <cstdio>
#include <string>

#include "json.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

sigil::json::Value parsed(const std::string& text, bool& ok) {
  sigil::json::Value v;
  std::string error;
  ok = sigil::json::parse(text, v, error);
  return v;
}

std::string quoted(const std::string& s) {
  std::string out;
  sigil::json::append_string(out, s);
  return out;
}

}  // namespace

int main() {
  bool ok = false;

  auto v = parsed(R"({"id":7,"op":"translate","keys":["en_ja"],"text":"a\"b\\c\n","priority":"final"})", ok);
  check(ok && v.get_u64("id", 0) == 7 && v.get_string("op") == "translate", "a request parses");
  check(v.get("text") && v.get("text")->string == "a\"b\\c\n", "escapes decode");
  check(v.get("keys") && v.get("keys")->array.size() == 1, "arrays parse");

  v = parsed(R"({"t":"\ud83d\ude00 \u65e5"})", ok);
  check(ok && v.get_string("t") == "\xF0\x9F\x98\x80 \xE6\x97\xA5", "surrogate pairs and BMP escapes become UTF-8");
  v = parsed(R"({"t":"\ud83d x \ude00"})", ok);
  check(ok && v.get_string("t") == "\xEF\xBF\xBD x \xEF\xBF\xBD", "lone surrogates become U+FFFD");

  v = parsed("{\"t\":\"\xFF ok\"}", ok);
  check(ok && v.get_string("t") == "\xEF\xBF\xBD ok", "invalid UTF-8 in input becomes U+FFFD");

  v = parsed(R"({"id":1,"id":2})", ok);
  check(ok && v.get_u64("id", 0) == 2, "the last duplicate key wins");
  v = parsed(R"({"id":-1})", ok);
  check(ok && v.get_u64("id", 99) == 99, "negative ids are not ids");
  v = parsed(R"({"id":1.5})", ok);
  check(ok && v.get_u64("id", 99) == 99, "fractional ids are not ids");
  v = parsed(R"({"id":18446744073709551616})", ok);
  check(ok && v.get_u64("id", 99) == 99, "out-of-range ids are not ids");

  parsed(R"({"a":1} x)", ok);
  check(!ok, "trailing data is refused");
  parsed(R"({"a":1,})", ok);
  check(!ok, "trailing commas are refused");
  parsed("{\"a\":\"line\nbreak\"}", ok);
  check(!ok, "raw control characters in strings are refused");
  parsed(std::string(100, '[') + std::string(100, ']'), ok);
  check(!ok, "deep nesting is refused");
  parsed("", ok);
  check(!ok, "empty input is refused");

  check(quoted("a\"\\\n\t\x01") == "\"a\\\"\\\\\\n\\t\\u0001\"", "strings are escaped");
  check(quoted("\xE2\x80\xA8") == "\"\\u2028\"", "U+2028 is escaped");
  check(quoted("\xC3\xA9\xF0\x9F\x98\x80") == "\"\xC3\xA9\xF0\x9F\x98\x80\"", "valid UTF-8 passes through");
  check(quoted("\xC0\xAF") == "\"\xEF\xBF\xBD\xEF\xBF\xBD\"", "overlong UTF-8 is replaced");
  check(sigil::json::sanitize_utf8("ok\xED\xA0\x80") == "ok\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "encoded surrogates are replaced");

  const std::string reply = sigil::json::Object().num("id", 3).boolean("ok", true).strings("loaded", {"en_ja", "ja_en"}).done();
  check(reply == R"({"id":3,"ok":true,"loaded":["en_ja","ja_en"]})", "the reply builder writes compact JSON");

  if (failures == 0) std::puts("json tests: ok");
  return failures == 0 ? 0 : 1;
}
