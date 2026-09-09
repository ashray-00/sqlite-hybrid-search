#pragma once

#include <cstdint>
#include <ctime>

// A single named call site for "what time is it", so every place that
// needs "now" for a timestamp agrees on how to get it. Internal
// implementation detail, not part of the public API.
namespace retrieval_engine::detail {

// Current wall-clock time as Unix epoch seconds. std::time() already
// returns seconds since 1970-01-01 UTC regardless of the system's local
// timezone setting -- there is no separate "UTC vs local" concern here,
// unlike std::localtime()/std::mktime(), which this code deliberately
// never uses.
inline std::int64_t CurrentUnixTimeSeconds() { return static_cast<std::int64_t>(std::time(nullptr)); }

}  // namespace retrieval_engine::detail
