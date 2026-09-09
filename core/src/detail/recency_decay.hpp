#pragma once

// Pure exponential recency decay math: no SQLite or usearch dependency,
// independently unit-testable. Internal implementation detail, not part of
// the public API.
namespace retrieval_engine::detail {

// Exponential recency decay for the agent memory layer:
//     Recency_Factor(age_seconds) = e^(-decay_lambda * age_seconds / 86400)
// normalizing age to days so decay_lambda values on the order of 0.05-0.5
// correspond to meaningful day-scale half-lives (rather than needing
// vanishingly small lambdas if age were left in raw seconds).
//
// Guarantees (independent review, Stage 4 GREEN->REVIEW pass):
//   - Always returns a finite value in [kMinRecencyFactor, 1.0]; never NaN
//     or Inf, regardless of input.
//   - A negative `age_seconds` (system clock stepped backwards between a
//     chunk's created_at and "now", or a caller-supplied created_at that is
//     in the future) is treated as zero age -- it can never *boost* a score
//     above its undecayed fused_score, only leave it undecayed.
//   - The result never drops below kMinRecencyFactor, however large
//     `age_seconds` or `decay_lambda` is: a sufficiently old or aggressively
//     decayed memory is discounted almost to nothing, but a caller-critical
//     old memory is never scored as if it did not exist (fully zeroed out).
double ComputeRecencyFactor(double age_seconds, double decay_lambda);

// The floor every recency factor is clamped to -- see ComputeRecencyFactor's
// contract above. Exposed so callers/tests can refer to the same constant
// rather than duplicating the magic number.
inline constexpr double kMinRecencyFactor = 0.01;

}  // namespace retrieval_engine::detail
