#include "sqlite_util.hpp"

#include <sstream>
#include <stdexcept>

namespace retrieval_engine::detail {

void ThrowIfSqliteError(int sqlite_rc, sqlite3* db, const std::string& context) {
    if (sqlite_rc == SQLITE_OK) return;
    std::ostringstream message;
    message << context << ": " << (db ? sqlite3_errmsg(db) : "unknown SQLite error");
    throw std::runtime_error(message.str());
}

namespace {

// A busy_timeout large enough that a reader connection never gives up while
// a writer holds the WAL lock for a normal ingest, small enough to surface
// a genuine deadlock/stuck writer rather than hang forever.
constexpr int kBusyTimeoutMs = 5000;

void ExecPragma(sqlite3* db, const char* pragma) {
    ThrowIfSqliteError(sqlite3_exec(db, pragma, nullptr, nullptr, nullptr), db,
                       std::string("SqliteConnection: failed to run ") + pragma);
}

}  // namespace

SqliteConnection::SqliteConnection(const std::string& db_path, int flags) {
    const int rc = sqlite3_open_v2(db_path.c_str(), &db_, flags, nullptr);
    // sqlite3_open_v2() assigns *db_ via the out-param even on failure (per
    // SQLite's own docs: a handle is returned so the caller can read the
    // error message off it), so db_ is already set here regardless of rc --
    // ~SqliteConnection() will close it correctly either way.
    ThrowIfSqliteError(rc, db_, "SqliteConnection: failed to open database at '" + db_path + "'");

    ExecPragma(db_, ("PRAGMA busy_timeout = " + std::to_string(kBusyTimeoutMs)).c_str());
    if ((flags & SQLITE_OPEN_READONLY) == 0) {
        // Read-write connection: put the database into WAL so pooled reader
        // connections can run concurrently with this writer.
        ExecPragma(db_, "PRAGMA journal_mode = WAL");
        ExecPragma(db_, "PRAGMA synchronous = NORMAL");
    }
}

SqliteConnection::~SqliteConnection() {
    if (db_) sqlite3_close(db_);
}

SqliteStatement::SqliteStatement(sqlite3* db, const char* sql) {
    ThrowIfSqliteError(sqlite3_prepare_v2(db, sql, -1, &statement_, nullptr), db,
                       std::string("failed to prepare statement: ") + sql);
}

SqliteStatement::~SqliteStatement() { sqlite3_finalize(statement_); }

std::string ColumnText(sqlite3_stmt* statement, int column) {
    const unsigned char* text = sqlite3_column_text(statement, column);
    return text ? std::string(reinterpret_cast<const char*>(text)) : std::string();
}

}  // namespace retrieval_engine::detail
