#include "recency_decay.hpp"

#include <algorithm>
#include <cmath>

namespace retrieval_engine::detail {

namespace {
constexpr double kSecondsPerDay = 86400.0;
}  // namespace

double ComputeRecencyFactor(double age_seconds, double decay_lambda) {
    // Clock skew / future-timestamp guard: a negative delta (system clock
    // stepped backwards, or a caller-supplied created_at that turns out to
    // be in the future) must never produce a factor above 1.0 -- clamp the
    // age used for the exponent, not the caller's raw age_seconds (callers
    // such as SearchExplanation report the true, unclamped age for
    // debugging/transparency; only the decay math itself treats it as 0).
    const double non_negative_age_seconds = std::max(age_seconds, 0.0);

    const double exponent = -decay_lambda * non_negative_age_seconds / kSecondsPerDay;
    const double factor = std::exp(exponent);

    // Numerical-stability guard: a pathological decay_lambda (very large, or
    // negative -- e.g. a caller mistakenly treating it as a "boost" knob)
    // combined with a very large age can overflow std::exp() to +Inf, or
    // underflow towards 0/NaN at the extremes. std::clamp()'s behavior is
    // unspecified when one of its arguments is NaN, so this must be checked
    // before clamping, not folded into it.
    if (!std::isfinite(factor)) {
        return kMinRecencyFactor;
    }

    // Floor so an old (or aggressively decayed) memory is heavily
    // discounted but never scored as if it did not exist at all; ceiling so
    // a factor can never *inflate* a score above its undecayed value.
    return std::clamp(factor, kMinRecencyFactor, 1.0);
}

}  // namespace retrieval_engine::detail
