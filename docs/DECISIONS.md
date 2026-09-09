# Blockers & Pain Points

Decision log across stages. Kept separate from BUILD_PLAN.md so the plan
stays a plan and this stays a record of what actually happened building it.

## Stage 3

### Decision: Python Engine API adapted to existing C++ functionality, not invented fresh

**What happened:** the Stage 3 RED instructions specified
`Engine(db_path, index_path)` and `engine.add(documents, embeddings)`.
Neither matches the real `RetrievalEngine` (core/include/retrieval_engine/
retrieval_engine.hpp) as it stands after three stages of refactoring:
- No `index_path`/file-based index persistence exists at all -- the
  architecture has always been SQLite + rebuild-the-usearch-sidecar-on-open
  (BUILD_PLAN.md section 5), reinforced across every stage's review.
  `dim` is required by the real constructor and wasn't in the given
  signature.
- `add_documents()` takes structured `DocumentInput`/`DocumentChunkInput`
  (pre-chunked, pre-embedded, with metadata and token offsets), not a flat
  `(documents, embeddings)` pair.

Flagged this to the user with two options (match the literal signature by
building new index-to-file C++ persistence, vs. adapt the Python surface to
what already exists). The user's answer: re-check the actual code (since
heavy refactoring has happened across stages) and confirm the *functionality*
is present even if names/shapes differ, rather than assuming a gap.
Re-read the header fresh and confirmed: `search_hybrid()`/`search_explained()`
already match name-and-shape exactly; only the constructor and `add()` had
real gaps, and both close via binding-layer adaptation, not new engine
capability.

**Resolved Python API (specified in tests/test_stage3.py):**
- `Engine(db_path: str, dim: int)` -- mirrors `RetrievalEngine(db_path, dim)`
  exactly; no `index_path`.
- `engine.add(documents, embeddings)` -- kept the literal two-parameter
  shape from the ask, but maps each `(document, embedding)` pair onto one
  `DocumentInput` with a *single* `DocumentChunkInput` (the whole document
  text as one chunk). Multi-chunk ingestion via `chunk_text()` already has
  its own coverage from Stage 1; this smoke test isn't the place to
  duplicate it.
- `engine.search(query_vec, top_k)` -> `search_dense()`.
- `engine.search_hybrid(...)` / `engine.search_explained(...)` -> unchanged,
  already matched.

**Revisit when:** if a real need for file-based index persistence emerges
(e.g. avoiding the rebuild-from-SQLite cost for very large corpora on
process start), that's new core C++ work belonging to its own stage/task --
matching BUILD_PLAN.md's own `.save()`/`.load()` bullet as methods, not a
constructor parameter -- not something to fold into Stage 3's bindings work.

### Decision: layered bindings -- a thin nanobind layer close to the C++ API, ergonomics in pure Python

`bindings/python_bindings.cpp` exposes `NativeEngine` (the raw
`RetrievalEngine`) and the structured `DocumentInput`/`DocumentChunkInput`/
`ChunkSearchResult`/`SearchExplanation` types fairly directly -- no dict
handling, no ingestion-shape adaptation in C++ at all. The public
`retrieval_engine.Engine` (`python/retrieval_engine/__init__.py`) is a pure-
Python wrapper that does the ergonomic work: converts the `(documents,
embeddings)` shape into native `DocumentInput` objects, and converts
`ChunkSearchResult`/`SearchExplanation` results into plain dicts. Kept the
adaptation logic in ordinary, easily-read/tested Python rather than manual
`nb::dict` handling in C++, which nanobind supports but which is
meaningfully more ceremony for the same result.

### Decision: pip-installed CLI needs *some* embedding to feed the dense index -- used a placeholder, documented loudly

`engine ingest`/`engine query` need an embedding per chunk to populate/query
`NativeEngine`'s dense index at all, but no embedding model exists yet
(BUILD_PLAN.md's own scope guard: "Start with caller-supplied embeddings...
add a built-in embedder in a later stage" -- Stage 5). Implemented
`_hash_embed()` (cli.py): a deterministic feature-hashing "bag of hashed
tokens" vector -- a real, if weak, technique (the "hashing trick"), not
random noise, so texts sharing words with a query get *some* dense-
similarity signal. Documented as a stand-in in the module docstring, the
function's own docstring, and the README, specifically so nobody mistakes
CLI search quality for a claim about the finished engine. Sparse (BM25) is
unaffected -- it works over literal text regardless of embeddings.

**Revisit when:** Stage 5 adds a real embedding model -- `_hash_embed()`
should be replaced (or become an explicit opt-in fallback), not left as the
silent default.

### Decision: CLI index lives in the current working directory, not the ingested folder

`engine query "<text>"` takes no folder argument (matching the literal
Stage 3 ask), so it must find the same database `engine ingest <folder>`
wrote without being told which folder was ingested. Both commands resolve
the database path via `Path.cwd()` (a fixed filename,
`.retrieval_engine.sqlite3`, not inside the ingested folder) -- `ingest`
reads *from* the given folder but writes its index into the current
directory, and `query` reads that same current-directory index. Caught by
tracing the RED test's exact `subprocess.run(..., cwd=tmp_path)` calls
before writing the GREEN implementation: an earlier draft wrote the index
inside the ingested folder itself, which would have made `query` (with no
folder argument) unable to find it.

### Review findings (Phase 3, independent review)

Focused on the three requested areas plus a general pass:

**Fixed (GIL):** `NativeEngine`'s constructor had no `nb::call_guard<nb::gil_scoped_release>()`,
unlike every other potentially-slow method on the class. Opening an
existing database rebuilds the *entire* dense index from every persisted
chunk (the same rebuild-on-open work reviewed back in Stage 1), which is
unbounded work for a large corpus -- inconsistent to guard `add_documents()`/
`search_*()` but not the constructor that can do just as much work. Fixed
by adding the same guard to `nb::init<...>()`.

**Fixed (CLI resilience):** `engine query --top-k -5` crossed a negative
Python int into `search_hybrid()`'s unsigned C++ `size_t` parameter.
Verified directly rather than assumed: nanobind's own caster correctly
*rejects* the negative value (no silent wraparound, no crash) but with a
raw type-mismatch message naming the compiled extension module and its
argument types -- not something a CLI user could act on. Fixed by
validating `--top-k > 0` in the CLI itself before calling into the engine,
with a regression test (`test_cli_query_rejects_non_positive_top_k`)
asserting the leaked message no longer appears.

**Fixed (documentation, preventive):** `DocumentInput.chunks` is a property
backed by nanobind's stl/vector.h get/set caster, not a live reference into
the C++ vector -- `doc.chunks.append(x)` would silently mutate a throwaway
temporary and lose the data, with no error at all. Nothing in the codebase
currently does this (every caller already does the correct
`doc.chunks = [...]` whole-list assignment), but it's a well-known nanobind/
pybind11 trap worth a comment at the binding site so a future contributor
doesn't rediscover it by losing data silently.

**Reviewed, no defect found:** vector/string conversions and memory-leak
risk generally (nanobind's own STL casters are used throughout with no
manual memory management anywhere in the binding file; the class's
non-copyable-but-movable design is fully compatible with nanobind's
placement-construct-in-place model, requiring neither copy nor move for
`nb::init<...>()`); five other CLI resilience scenarios (nonexistent
folder, a file passed where a directory is expected, an empty folder,
querying before any ingest, no subcommand at all) -- all five verified by
direct execution to already produce clean, correctly-exit-coded errors.

## Stage 2

### Decision: split ChunkStore into 4 components + a thin orchestrator, renamed ChunkSearchResult::distance to score (user-requested)

**What happened:** by the end of Stage 2, `chunk_store.cpp` had grown to 440
lines covering four distinct concerns (SQLite persistence for
documents/chunks/chunks_fts, the usearch dense index, FTS5 query
sanitization, RRF fusion math) bundled into one class. The user asked
directly whether it needed splitting along SOLID lines and whether any
names needed improving.

**Options considered for the split:** (1) extract only the two pure,
storage-agnostic pieces (RRF math, FTS5 sanitization) and leave persistence
(schema + add_documents + search_dense/search_sparse) together in
ChunkStore, since the write path's atomicity across chunks/chunks_fts/
usearch is a real, load-bearing reason to keep it coupled; (2) a full
4-component split.

**Decision (chosen by the user):** option 2. Resulted in:
- `ChunkRepository` -- SQLite persistence: schema for *all three* of
  documents/chunks/chunks_fts (grouped together deliberately: all three are
  plain, durable, transactional SQLite constructs written atomically in one
  `add_documents()` transaction), plus `Resolve()` and `SearchSparse()`.
- `DenseIndex` -- a thin usearch wrapper with zero SQLite dependency, only
  integer keys and vectors in and out.
- `fts5_query.{hpp,cpp}` -- `BuildSafeFts5MatchQuery()` as a pure free
  function (not a class: it's stateless data transformation, matching the
  existing `sqlite_util`/`usearch_util` free-function pattern rather than
  forcing OOP ceremony onto logic that doesn't need identity or state).
- `rrf_fusion.{hpp,cpp}` -- `RrfFuse()`, also a pure free function, named to
  match BUILD_PLAN.md's own literal `rrf_fuse(dense_results, sparse_results)`
  naming.
- `ChunkStore` -- reduced to a 107-line orchestrator: composes
  `ChunkRepository` + `DenseIndex`, validates dimensions up front, and
  enforces the "populate DenseIndex only after ChunkRepository's SQLite
  transaction commits" ordering (the Stage 1 review finding) at the
  coordination layer instead of inline.

**Interface note:** `ChunkRepository::ForEachChunk()` (used to rebuild
`DenseIndex` on open) hands the visitor an owned `std::vector<float>` per
chunk rather than a raw pointer into the SQLite blob, which the prior
combined implementation used directly with zero extra copying. This adds
one extra copy per chunk during rebuild-on-open only (not on the
`add_documents()` ingestion path BUILD_PLAN.md's "10k chunks" target
concerns). Chose the simpler, uniform `vector<float>`-everywhere interface
over a zero-copy visitor/raw-pointer scheme: at realistic scale (10k chunks
x a few hundred floats each) the extra copy is on the order of tens of
milliseconds, once, at process startup if reopening an existing database --
not worth the interface complexity to avoid.

**Also fixed:** `ChunkSearchResult::distance` renamed to `score`.
`distance` wrongly implied "lower is better" universally, but
`search_hybrid()`'s fused RRF score is higher-is-better -- the opposite
convention, in a field that had to apologize for its own name in its doc
comment. Breaking change to the public header, done now because Stage 3's
Python bindings don't exist yet to make it expensive later. No test
directly read the field, so this needed no test changes.

Pure refactor otherwise: all 15 tests passed unmodified throughout, zero
warnings under `-Wall -Wextra -Wpedantic`, verified with a clean
`rm -rf build` cycle.

### Checked: FTS5 is compiled into the linked SQLite3 (it is -- not a blocker)

FTS5 is an optional, compile-time SQLite feature -- not guaranteed by every
build, unlike the core library. Before writing the Stage 2 RED test, verified
it against the *exact* linked library (`/opt/homebrew/opt/sqlite`), not just
the `sqlite3` CLI (which can be built with different flags than the
library): compiled a minimal `CREATE VIRTUAL TABLE ... USING fts5(...)`
smoke test against our actual include/lib paths. It works
(`ENABLE_FTS5` is in Homebrew sqlite 3.53.4's `compile_options`). Kept as a
permanent regression test (`FtsLibrary.RanksRowsByBm25Directly` in
test_stage2.cpp) rather than a one-off check, mirroring the
usearch/SQLite3 sanity tests from Stage 0.

**Revisit when:** if the project ever moves off Homebrew's sqlite (e.g. a
vendored/FetchContent'd SQLite amalgamation for portability, similar to how
usearch is vendored), re-verify FTS5 is enabled in that build too -- it's
not implied by "SQLite3 found" the way it was implicitly assumed here.

### Decision: added search_dense() alongside search_chunks() rather than renaming (deferred to GREEN)

Stage 2's spec names a `search_dense()` method to pair with the new
`search_sparse()`/`search_hybrid()`, but Stage 1 already shipped
`search_chunks()` doing the exact same cosine-similarity search. Declared
`search_dense()` as a new method for this RED pass rather than renaming
`search_chunks()` in place, so Stage 1's own tests (which call
`search_chunks()`) don't need to change as a side effect of Stage 2's RED
phase -- consistent with the pattern already used for
`search`/`search_chunks` in Stage 0->1.

**Revisit at Phase 2 (GREEN):** decide whether `search_chunks()` becomes a
thin alias for `search_dense()`, or is retired and Stage 1's tests updated
to call `search_dense()` directly (mirroring how Stage 0's dummy API was
retired once Stage 1 shipped its real replacement). Shipping both as
independent, duplicate implementations would be the wrong outcome.

**Resolved at GREEN:** retired `search_chunks()` outright and renamed the
implementation to `search_dense()`; updated Stage 1's tests
(`test_stage1.cpp`) to call `search_dense()` directly. No alias kept -- the
two names would have been 100% identical in behavior forever, unlike Stage
0's dummy API which was genuinely different data. Zero other callers existed
(no Python bindings/CLI yet), so nothing else needed updating.

### Blocker: search_sparse() didn't sanitize query_text against FTS5's query grammar

**What happened:** caught in the Phase 3 independent review (the explicit
"FTS5 query string sanitization" focus area). `search_sparse()`'s first
GREEN implementation bound raw `query_text` straight into FTS5's MATCH
operand. MATCH isn't a literal string -- it's parsed by FTS5's own query
grammar (AND/OR/NOT, a leading `-` meaning NOT, quoted phrases, `column:`
filters, `*` prefix queries). Verified against the *exact* linked SQLite3
(not assumed): `search_sparse("ZXQ7742 -stock", ...)` raised `no such
column: stock`, and an unbalanced quote raised `unterminated string` --
both entirely ordinary real-world search input (a hyphenated product code,
a stray quote) throwing instead of matching literal text.

**Fix:** `BuildSafeFts5MatchQuery()` splits the input on whitespace and
wraps each token in its own double-quoted phrase (escaping embedded `"` by
doubling -- confirmed against the linked SQLite3 that this is FTS5's actual
escape convention, not assumed), OR'd together. A quoted phrase is
tokenized like ordinary text, never parsed for operators, so nothing the
caller types can be interpreted as query syntax. OR (rather than implicit
AND) matches typical keyword-search UX: find chunks containing *any* of the
given terms. Added a regression test
(`SearchSparseTreatsSpecialCharactersAsLiteralText`) covering the hyphen,
the unbalanced quote, and an all-whitespace query.

**Also reviewed, no defect found:** the 1-based rank arithmetic throughout
RRF fusion (verified against the `1/(k+rank)` formula line by line), and
transaction integrity for the `chunks`/`chunks_fts` dual insert (both live
inside the same `BEGIN`/`COMMIT`/`ROLLBACK`, `chunk_id` captured once and
reused explicitly rather than re-queried). One MINOR fix made anyway while
in the area: added an explicit tie-breaker (`document_id`, then
`chunk_index`) to RRF's final sort, since leaving equal-fused-score ties to
`std::sort`'s unspecified ordering undermines the determinism an
explainability feature should provide.

## Stage 1

### Decision: replaced istringstream with manual scanning in chunk_text() (user-requested)

**What happened:** the user asked whether `istringstream` and other STL
choices across the codebase were sound for memory/performance, citing
well-documented iostream criticism. Audited every `stringstream`/`ostringstream`
use in the codebase: `sqlite_util.cpp`'s `ThrowIfSqliteError` builds one on
an error path only (fires on failures, never in a loop -- not worth
touching, an exception already costs far more), but `chunking.cpp`'s
`chunk_text()` used `istringstream` for tokenization on what is a genuine
hot path -- called once per document ingested, with BUILD_PLAN.md's own
Stage 1 scale target being "ingestion of 10k chunks". `istringstream` pays
for generality unused here: a global-locale lookup on construction and a
virtual call through its streambuf per character extracted.

**Fix:** manual `find_first_not_of`/`find_first_of` scanning producing
`std::string_view` tokens (no per-token allocation during tokenization),
plus `reserve()`-ing each chunk's text before the `+=` loop that assembles
it. Behavior-preserving -- all 10 tests passed unmodified throughout,
including the exact token-boundary and empty-input cases, used as the
correctness safety net rather than re-deriving it by inspection.

**Also reviewed and left alone:** every other `std::string` concatenation in
the codebase (`chunk_store.cpp`, `sqlite_util.cpp`) only happens immediately
before `throw` on an error path -- not a hot path, so not worth the same
treatment. No `std::regex`, `std::endl`, or other commonly-flagged STL
performance pitfalls were found elsewhere in the codebase.

### Decision: split retrieval_engine.cpp into focused modules (user-requested)

**What happened:** by the end of Stage 1, `retrieval_engine.cpp` had grown
to 408 lines covering three unrelated concerns (generic SQLite RAII helpers,
Stage 0's dummy-vector scaffolding, Stage 1's real chunk store) and `Impl`
was accumulating state for both. The user asked directly whether this should
be split up along SOLID lines.

**Options considered:** (1) full split -- generic SQLite helpers, each
store, and the facade all separated into their own files; (2) Stage-1-only
split, leaving Stage 0's code where it was to avoid touching already-shipped
code; (3) extract just the generic SQLite helpers and leave both stores'
logic inline.

**Decision (chosen by the user):** option 1. Resulted in
`sqlite_util.{hpp,cpp}` (generic), `usearch_util.hpp` (generic, header-only),
`dummy_vector_store.{hpp,cpp}` (Stage 0, isolated so it's easy to delete
later), `chunk_store.{hpp,cpp}` (Stage 1's real feature), and
`retrieval_engine.cpp` reduced to a 63-line thin Pimpl facade delegating to
the two stores. Deliberately did not force a shared interface between the
two stores (would satisfy Liskov/DIP on paper but neither caller nor test
treats them polymorphically today -- abstraction without a present need).
Pure refactor: all 13 tests passed unmodified throughout, used as the safety
net rather than re-deriving correctness by inspection.

**Incidental improvement:** introduced a `SqliteConnection` RAII type as
part of the split, replacing the old "assign `impl_->db` before throwing"
manual trick that earlier stages relied on for leak-safety. Opening the
connection is now just another initializer-list member, so a later store's
constructor throwing during `Impl` construction closes the connection via
ordinary member-destruction-on-exception rules, no manual sequencing needed.

**Revisit when:** Stage 2 (hybrid/FTS5) or Stage 4 (memory) add their own
stores -- confirm they follow the same pattern (own table(s), own index if
any, constructed from a non-owning `sqlite3*`) rather than growing `Impl` or
an existing store.

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

**Resolved:** the user asked directly whether the dummy scaffolding was
still needed. Removed it: `add_vector`/`search`/`dummy_table_row_count`,
`dummy_vectors`, and the whole `dummy_vector_store.{hpp,cpp}` file (added
during the SOLID split above) are gone. Nothing else in the codebase
depended on them -- no Python bindings or CLI exist yet (Stage 3), and
`test_stage0.cpp`'s `RetrievalEngineStage0.*` tests were the only consumers,
also removed. Its two tests that exercised usearch/SQLite3 directly
(independent of the dummy API) were kept and moved to
`test_infra_sanity.cpp`; the still-relevant `ConstructorRejectsZeroDimension`
test (that validation lives in `RetrievalEngine`'s constructor itself, not
in either store) moved to `test_stage1.cpp`. All 10 remaining tests pass
unmodified in content, zero warnings under `-Wall -Wextra -Wpedantic`.

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
