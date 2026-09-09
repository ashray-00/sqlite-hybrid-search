#pragma once

#include "sqlite_util.hpp"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct sqlite3;

// A fixed pool of read-only SQLite connections to one database file, so
// concurrent readers run against a WAL snapshot in parallel instead of
// serialising on the single writer connection's internal mutex (ADR-11).
//
// Inert for databases with no stable path (":memory:" / "") -- a second
// connection would open a *different* in-memory database. There, Acquire()
// hands back the writer connection and reads serialise as before, which is
// fine for tests.
namespace retrieval_engine::detail {

class ReadConnectionPool {
public:
    // `db_path` is the SQLite file. If empty or ":memory:" the pool is
    // inert and every Acquire() returns `fallback_connection`. Otherwise
    // opens `size` SQLITE_OPEN_READONLY connections up front. `size` is
    // clamped to >= 1.
    ReadConnectionPool(const std::string& db_path, sqlite3* fallback_connection, std::size_t size);

    ReadConnectionPool(const ReadConnectionPool&) = delete;
    ReadConnectionPool& operator=(const ReadConnectionPool&) = delete;

    // RAII checkout. get() is valid for the handle's lifetime; the
    // connection returns to the pool on destruction. Move-only.
    class Handle {
    public:
        Handle(ReadConnectionPool* pool, sqlite3* connection) : pool_(pool), connection_(connection) {}
        Handle(Handle&& other) noexcept : pool_(other.pool_), connection_(other.connection_) {
            other.pool_ = nullptr;
            other.connection_ = nullptr;
        }
        Handle& operator=(Handle&&) = delete;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        ~Handle();

        sqlite3* get() const { return connection_; }

    private:
        ReadConnectionPool* pool_;  // nullptr => inert handle, nothing to return
        sqlite3* connection_;
    };

    // Blocks until a connection is free (bounded by `size` concurrent
    // holders). Inert pools never block.
    Handle Acquire();

private:
    void Return(sqlite3* connection);

    std::vector<std::unique_ptr<SqliteConnection>> connections_;  // owned; empty when inert
    std::vector<sqlite3*> free_;
    std::mutex mutex_;
    std::condition_variable available_;
    sqlite3* fallback_connection_;
    bool inert_;
};

}  // namespace retrieval_engine::detail
