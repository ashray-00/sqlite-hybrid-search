#pragma once

#include <sqlite3.h>

#include <string>

// Small, generic SQLite helpers shared by every store in this library. None
// of this is retrieval_engine-specific -- it would be equally at home
// wrapping any other SQLite-backed component (the FTS5 sparse index, a
// future memory store). Internal implementation detail, not part of the
// public API: colocated with its .cpp in core/src/, not under include/.
namespace retrieval_engine::detail {

// Throws std::runtime_error naming `context` and SQLite's own error message
// unless `sqlite_rc` is SQLITE_OK. Only meant for calls (open/prepare/exec)
// whose success code is SQLITE_OK -- sqlite3_step()'s SQLITE_DONE/SQLITE_ROW
// mean something different and are checked at their call sites instead.
void ThrowIfSqliteError(int sqlite_rc, sqlite3* db, const std::string& context);

// Owns a SQLite connection: opens it in the constructor, closes it in the
// destructor. Exists so a constructor that opens a connection and then does
// more (fallible) work -- creating tables, rebuilding an index -- doesn't
// need a manual "assign before throw" dance to stay leak-safe: as long as
// this is a member initialized before anything that can throw, ordinary
// member-destruction-on-exception rules close it automatically.
//
// Every connection sets a busy_timeout so a concurrent writer never
// surfaces SQLITE_BUSY to a reader. A read-write connection additionally
// switches the database to WAL journalling (persisted in the file header,
// so read-only connections pick it up) with synchronous=NORMAL -- WAL lets
// many reader connections run against a snapshot while one writer commits
// (ADR-11). `:memory:` databases ignore WAL harmlessly.
class SqliteConnection {
public:
    // `flags` are passed to sqlite3_open_v2(); the default opens (creating
    // if absent) read-write. Pass SQLITE_OPEN_READONLY for a pooled reader.
    explicit SqliteConnection(const std::string& db_path, int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    ~SqliteConnection();

    SqliteConnection(const SqliteConnection&) = delete;
    SqliteConnection& operator=(const SqliteConnection&) = delete;

    sqlite3* get() const { return db_; }

private:
    sqlite3* db_ = nullptr;
};

// RAII wrapper for a prepared sqlite3_stmt*. Methods with multiple throw
// points (validation, usearch failures) between preparing a statement and
// finishing with it need this -- tracking sqlite3_finalize() by hand across
// all of them is exactly the kind of thing that silently leaks a statement
// handle on one missed path. Reset-and-rebind for repeated use (e.g. once
// per row in a loop) via .get().
class SqliteStatement {
public:
    SqliteStatement(sqlite3* db, const char* sql);
    ~SqliteStatement();

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;

    sqlite3_stmt* get() const { return statement_; }

private:
    sqlite3_stmt* statement_ = nullptr;
};

// sqlite3_column_text() can return nullptr (a SQL NULL, or on OOM); building
// a std::string from a null `const char*` is undefined behaviour. Guard it
// even where the schema declares the column NOT NULL, per CLAUDE.md's
// "defensive programming at boundaries".
std::string ColumnText(sqlite3_stmt* statement, int column);

}  // namespace retrieval_engine::detail
