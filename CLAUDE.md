# Environment & Architecture
- Platform: macOS ARM64 (Apple Silicon M1/M2/M3)
- Homebrew Prefix: `/opt/homebrew` (Include Path: `/opt/homebrew/include`, Lib Path: `/opt/homebrew/lib`)

# Project Overview
Embeddable Retrieval & Memory Engine written in C++17 with Python bindings (nanobind) and SQLite + usearch integration.

# Command Shortcuts
- Setup venv: already done
- C++ Build: `cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build build`
- C++ Tests: `ctest --test-dir build --output-on-failure`
- Install Python Bindings in venv: `.venv/bin/pip install -e .`
- Run Python Tests in venv: `.venv/bin/pytest`

# Project Directives
1. Virtual Environment Usage: ALL Python execution, installation, and testing MUST run through `.venv/bin/python`, `.venv/bin/pip`, or `.venv/bin/pytest`. Do NOT use global system Python.
2. M1 Architecture Compatibility: Ensure CMake includes `/opt/homebrew/include` and `/opt/homebrew/lib` so native Homebrew packages (like `sqlite3` or `libomp`) resolve cleanly on ARM64.
3. Single Stage Execution: NEVER attempt multiple stages at once. Focus strictly on the current stage defined in `BUILD_PLAN.md`.
4. Strict TDD Workflow: ALWAYS write a failing test FIRST (Red) before writing any production implementation (Green). Never write implementation and tests in the same pass.
5. Review Discipline: Apply the Author -> Verification -> Independent Reviewer -> Fix loop before declaring any step done.
6. If any library is missing while running say it is missing and will install it manually.
7. Blocker/Decision Logging: At the end of every run, append any blockers, upstream bugs, environment friction, or non-obvious decisions hit during that run to `docs/DECISIONS.md` (create it if missing). For each entry record: what happened, why, the options considered, which one was chosen (and by whom, if the user decided), and when to revisit it. Skip this only if the run hit no such issues.

---

# Test-Driven Development (TDD) Protocol

Every feature or bug fix must strictly follow the 3-phase TDD cycle:

### Phase 1: RED (Write Failing Tests First)
- Write unit/integration tests that describe the required behavior, interfaces, edge cases, and failure modes.
- Tests MUST fail when run because the implementation does not exist or is incomplete.
- Run `ctest` or `pytest` to VERIFY that the tests actually fail for the expected reason (e.g., missing symbol, assertion error). Do NOT write implementation code in this pass.

### Phase 2: GREEN (Minimal Implementation)
- Write the minimal production code necessary to satisfy the failing tests.
- Do NOT add extraneous features, extra abstractions, or future-proofing.
- Run `ctest` or `pytest` to confirm all tests pass cleanly.

### Phase 3: REFACTOR & REVIEW (Clean Up & Verify)
- Refactor the code for readability, performance, RAII/memory safety, and architecture adherence.
- Ensure all tests REMAIN green after refactoring.
- Run static analysis, linting, or compiler warning checks.

---

# Single-Stage Progress Enforcement
- Refer to `BUILD_PLAN.md` for stage definitions.
- You are forbidden from implementing requirements from Stage N+1 while working on Stage N.
- A stage is complete ONLY when all its specific goals are verified via green tests, and an independent review pass yields zero BLOCKER or IMPORTANT findings.

---

# Engineering Quality & Independent Review Principle

Treat every code change as if it were being developed by a professional software engineering team, not as a one-shot code generation task.

## 1. Understand Before Changing
Before modifying code:
- Understand existing architecture, conventions, abstractions, and dependencies.
- Inspect relevant code paths rather than making assumptions.
- Identify how proposed changes fit into the design.
- Reuse existing utilities, patterns, interfaces, tests, and abstractions.
- Determine expected behavior, edge cases, failure modes, and backwards-compatibility requirements.
- Prefer a small, well-understood change over a broad rewrite.

## 2. Design Before Implementation
Establish a clear approach considering:
- Separation of concerns & Single responsibility
- Appropriate abstraction boundaries & Encapsulation
- Resource ownership, memory safety (RAII in C++), lifetime, and thread safety
- Performance, security, testability, and maintainability
Use the simplest design that is robust and maintainable.

## 3. Implement Like a Senior Engineer
Write production-quality, clear, idiomatic code:
- Defensive programming at system boundaries.
- Explicit error handling and resource cleanup.
- Avoid unnecessary complexity, duplicated logic, magic values, dead code, global state, and suppressed compiler warnings.
- Comments must explain *why*, not merely restate *what* the code does.

## 4. Test the Change
- Follow the TDD Protocol (Tests written FIRST).
- Test happy paths, edge cases, and failure paths.
- Run the project's test suite (`ctest` / `pytest`).
- Verify tests pass with zero compiler or runtime warnings.

## 5. Perform an Independent Code Review
After implementation, STOP acting as the author. Perform a separate review as an independent senior engineer reviewing another developer's pull request.
- Review for: **Correctness**, **Architecture**, **Maintainability**, **Reliability**, **Concurrency**, **Performance**, **Security**, and **Test Sufficiency**.

## 6. Review Must Be Independent
Use this mental model: *"Assume the implementation contains at least one important mistake. My job is to find it."*

## 7. Create Review Findings
Classify findings:
- **BLOCKER**: Must be fixed before acceptance (e.g., memory leak, crash, incorrect logic, race condition, broken contract).
- **IMPORTANT**: Should normally be fixed before acceptance (e.g., missing edge-case test, poor error handling).
- **MINOR**: Worth improving but does not block acceptance (e.g., naming, minor style).

## 8. Author → Reviewer → Author Loop
1. **Phase A (Author - Red/Green TDD):** Implement test, then code.
2. **Phase B (Verification):** Build and test (`ctest`/`pytest`).
3. **Phase C (Independent Reviewer):** Critically review diff and log findings.
4. **Phase D (Fix):** Address `BLOCKER` and `IMPORTANT` findings.
5. **Phase E (Re-review):** Re-review fixes.

## 9. Never Review and Fix in the Same Mental Pass
Complete review, log findings, switch roles, fix code, re-verify, then start a fresh review pass.

## 10. Regression Protection
After fixing findings: re-run full test suite to ensure zero regressions.

## 11. Scope Discipline
Keep changes focused on the current stage and task. Do not refactor unrelated code unless necessary for the current task.

## 12. Definition of Done
A task/stage is **DONE** only when:
- Stage-specific requirements from `BUILD_PLAN.md` are satisfied.
- Tests were written FIRST and are passing (`ctest` green).
- Independent review was completed with all `BLOCKER` and `IMPORTANT` findings resolved.