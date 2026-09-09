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

SqliteConnection::SqliteConnection(const std::string& db_path) {
    const int rc = sqlite3_open(db_path.c_str(), &db_);
    // sqlite3_open() assigns *db_ via the out-param even on failure (per
    // SQLite's own docs: a handle is returned so the caller can read the
    // error message off it), so db_ is already set here regardless of rc --
    // ~SqliteConnection() will close it correctly either way.
    ThrowIfSqliteError(rc, db_, "SqliteConnection: failed to open database at '" + db_path + "'");
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
