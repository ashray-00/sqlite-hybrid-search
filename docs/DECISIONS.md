# Blockers & Pain Points

Decision log across stages. Kept separate from BUILD_PLAN.md so the plan
stays a plan and this stays a record of what actually happened building it.

## Stage 1

### Blocker: usearch mutations aren't transactional, so a naive add_documents() could desync the index from SQLite on a partial failure

**What happened:** the first GREEN implementation of `add_documents()` called
`chunk_index.add()` (usearch) inside the same loop as the SQLite row inserts,
before `COMMIT`. Caught in the Phase 3 independent review (prompted by the
explicit "correct usearch vector ID mapping to SQLite rowids" review focus):
if a later document in a multi-document batch fails -- e.g. two documents in
one call sharing a `document_id`, tripping the `documents.document_id`
PRIMARY KEY constraint -- the `catch` block issues `ROLLBACK`, correctly
undoing every SQLite row from the batch, but usearch has no equivalent
concept and keeps every vector already `.add()`'d for earlier chunks in that
same batch. That's a usearch entry with no backing SQLite row -- exactly the
inconsistency the "SQLite is authoritative, usearch is a rebuildable
sidecar" architecture (BUILD_PLAN.md section 5) has no recovery story for,
and the same class of bug Stage 0's review fixed for `add_vector()` (see
below), reintroduced in a form a lot easier to trigger (any multi-document
batch with a later failure).

**Fix:** defer every `chunk_index.add()` call until *after* SQLite's
transaction commits, buffering `(chunk_id, embedding)` pairs during the
transaction. A failure during the deferred usearch-population loop (after
commit) can only ever leave the index *lagging* what's in SQLite -- the
architecturally sanctioned, recoverable direction -- never containing an
entry SQLite has no record of. Added a regression test
(`RetrievalEngineStage1.FailedBatchLeavesNoPhantomEntriesInChunkIndex`) that
forces exactly this rollback path and asserts the index comes out empty
afterward, not desynced.

**Revisit when:** if a future stage wants ingestion to be resumable/streamed
(each document committed independently rather than one batch = one
transaction), this buffering approach needs re-examining -- per-document
transactions would need per-document index population immediately after each
document's own commit, not deferred to the end of the whole call.

### Decision: kept Stage 0's dummy_vectors/add_vector/search/dummy_table_row_count untouched

Stage 1 needed a `search()`-like method returning rich chunk results, but
Stage 0 already used that exact name (`search(vector, k) -> vector<uint64_t>`)
with an identical parameter list -- C++ can't overload on return type alone,
so reusing the name wasn't an option without changing Stage 0's shipped
contract. Named the new method `search_chunks()` instead and left Stage 0's
`add_vector`/`search`/`dummy_table_row_count`/`dummy_vectors` table
completely alone (still their own separate usearch index, `index`, distinct
from Stage 1's `chunk_index`). Purely additive: zero regression risk, but it
does mean the class now carries two parallel "add a vector" paths
permanently. Revisit in a later cleanup-focused stage -- retiring Stage 0's
scaffolding now that Stage 1's real schema exists would be a deliberate,
reviewed decision, not a side effect of adding Stage 1.

## Stage 0

## Blocker: usearch v2.9.2 fails to compile under AppleClang

**What happened:** every tagged usearch release we checked (`v2.9.2` down to
at least `v2.8.15`) fails to compile with AppleClang 21 (Xcode's current
compiler on this machine). The error surfaces as soon as any code calls
`index_dense_t::add()`:

```
error: member reference type 'const index_gt<...>' is not a pointer;
did you mean to use '.'?
    vector_key_t key() const noexcept { return index_->node_at_(slot()).key(); }
```

**Root cause:** in `include/usearch/index.hpp`, the nested class
`candidates_iterator_t` stores `index_gt const& index_` — a *reference* — but
`key()` dereferences it with `->` instead of `.`. That's a plain typo in
upstream usearch, not anything on our side. It's only hit when a code path
that constructs a `candidates_range_t`/`candidates_iterator_t` is actually
instantiated (duplicate-key scanning inside `add()`), which is why a
"header-only, just include it" sanity check wouldn't necessarily have caught
this ahead of time.

**Verification:** confirmed the same bug exists unchanged in `v2.9.2`,
`v2.9.1`, `v2.9.0`, `v2.8.16`, `v2.8.15` (fetched each tag's `index.hpp` from
GitHub and diffed the relevant line). Confirmed usearch's unreleased `main`
branch (commit `f91fe5b`, `VERSION` file says `2.26.2`) has already fixed it
to `index_.node_at_(slot()).key()` — i.e. this is a known-and-fixed-but-not
-yet-released bug, not a misunderstanding of the API on our part.

**Options considered:**
1. Pin `FetchContent` to the unreleased `main` commit (`f91fe5b`) that has
   the fix.
2. Stay on the citable `v2.9.2` tag and patch the vendored header ourselves
   at configure time, applying the exact same one-character fix upstream
   later shipped.

**Decision (chosen by the project owner):** option 2. Staying on a numbered
release is worth more than avoiding a patch step, given the patch is a single
verified character (`->` → `.`), matches upstream's own eventual fix
verbatim, and is guarded to fail loudly (`FATAL_ERROR`) if the text it
expects to find ever changes — so a future usearch bump can't silently apply
a stale patch to code that no longer needs it.

**Where it lives:** `cmake/patch_usearch_candidates_iterator_bug.cmake`,
invoked from the top-level `CMakeLists.txt` right after
`FetchContent_Populate(usearch)`.

**Revisit when:** usearch cuts a release newer than `v2.9.2` that includes
the fix — at that point bump `GIT_TAG` and delete the patch step.

---

## Other pain points (environment/tooling friction)

These didn't block the build but cost real time to diagnose, so recording
them for the next person (or future me) touching this CMake setup.

1. **Homebrew's `sqlite` is keg-only.** macOS ships its own (older) SQLite,
   so Homebrew doesn't link `sqlite` into `/opt/homebrew/include` or
   `/opt/homebrew/lib` by default — `find_package(SQLite3)` with only
   `-DCMAKE_PREFIX_PATH=/opt/homebrew` silently resolves to the system
   version instead of Homebrew's, unless something disambiguates. Fixed by
   shelling out to `brew --prefix sqlite` at configure time and appending
   that path to `CMAKE_PREFIX_PATH`, per CLAUDE.md's ARM64-compatibility
   directive.

2. **`SQLite::SQLite3` vs. `SQLite3::SQLite3`.** CMake's bundled
   `FindSQLite3` module now warns that the `SQLite::SQLite3` imported target
   name is deprecated in favor of `SQLite3::SQLite3`. Used the new name to
   keep the configure step warning-free.

3. **`FetchContent_Populate()` deprecation.** We want usearch's *sources*
   without `add_subdirectory()`-ing its own `CMakeLists.txt` (which pulls in
   C/Python/Rust/Go/Java/etc. bindings and their own dependencies we don't
   want). The direct `FetchContent_Populate()` call that does this is
   deprecated in favor of `FetchContent_MakeAvailable()`, which doesn't
   support skipping `add_subdirectory()` when a `CMakeLists.txt` is present.
   Resolved by explicitly setting `cmake_policy(SET CMP0169 OLD)` around the
   call, documented in-line so it isn't mistaken for an oversight later.

4. **usearch's `fp16`/`simsimd`/`stringzilla` git submodules.** usearch
   vendors hardware-acceleration helpers as submodules. On Apple Silicon
   with usearch's default build options (`USEARCH_USE_FP16LIB` resolves to
   `0` because ARM has native `__fp16`; `USEARCH_USE_SIMSIMD` defaults to
   `0`), neither is actually included by the headers we use. Declared
   `GIT_SUBMODULES ""` on the `FetchContent_Declare` to skip cloning them,
   keeping the configure step faster — flag this decision if a later stage
   turns on SIMSIMD or needs software fp16 emulation on non-ARM targets.

5. **No usearch or GoogleTest package in Homebrew.** Neither is available as
   a Homebrew formula, so both are pulled via `FetchContent` rather than
   `find_package`. This matches the Stage 0 ask ("fetch usearch via CMake
   FetchContent") but means every clean `build/` directory pays a one-time
   network-fetch + GoogleTest-compile cost (a few minutes) before the actual
   project builds.

6. **The usearch patch script wasn't idempotent across separate `cmake`
   invocations.** The `FetchContent_GetProperties`/`usearch_POPULATED` guard
   around the patch step only holds within a single `cmake` process; the
   internal reconfigure that `cmake --build` triggers when a `CMakeLists.txt`
   changes starts a fresh process where that guard variable is unset again,
   so it tried to re-apply the patch to an already-patched file and hit its
   own `FATAL_ERROR` safety check. Fixed by making the script check for the
   *fixed* text first and treat that as success, only erroring if neither the
   buggy nor the fixed text is present.

7. **usearch's "reserve capacity before adding" precondition fails silently
   (segfault, not an error).** `index_dense_gt::add()` expects
   `size() < capacity()` to already hold; skip the `reserve()` call and it
   crashes rather than returning a checkable `add_result_t` failure. This
   bit the Stage 0 test itself (the direct-usearch sanity test segfaulted
   until a `reserve()` call was added) and is now handled inside
   `RetrievalEngine::add_vector()` with amortized-doubling growth. Worth
   remembering for Stage 1's ingestion path, which will call `add()` in a
   loop over a much larger corpus.

8. **Decision: write SQLite before usearch in `add_vector()`.** The initial
   GREEN implementation added to the usearch index first, then inserted the
   SQLite bookkeeping row -- backwards from BUILD_PLAN.md section 5's stated
   architecture ("SQLite is authoritative, usearch is a rebuildable
   sidecar"). Caught in the Phase 3 independent review and reordered:
   SQLite write happens first, so a failure there leaves no orphaned vector
   in the index; if the later usearch add fails, the index merely lags what
   SQLite already durably recorded, which the architecture's own "rebuild
   from SQLite" story can recover from.
