#pragma once

#include "blocking_pool.hpp"
#include "sqlite_util.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

// A fixed pool of read-only SQLite connections to one database file, so
// concurrent readers run against a WAL snapshot in parallel instead of
// serialising on the single writer connection's internal mutex (ADR-11).
//
// Inert for databases with no stable path (":memory:" / "") -- a second
// connection would open a *different* in-memory database. There, Acquire()
// hands back the writer connection and reads serialise on it as before.
namespace retrieval_engine::detail {

class ReadConnectionPool {
public:
    // `db_path` is the SQLite file. If empty or ":memory:" the pool is
    // inert and every Acquire() returns `fallback_connection`. Otherwise
    // opens `size` (clamped to >= 1) SQLITE_OPEN_READONLY connections up
    // front.
    ReadConnectionPool(const std::string& db_path, sqlite3* fallback_connection, std::size_t size);

    ReadConnectionPool(const ReadConnectionPool&) = delete;
    ReadConnectionPool& operator=(const ReadConnectionPool&) = delete;

    // RAII checkout. get() is valid for the handle's lifetime; the
    // connection returns to the pool on destruction. Move-only.
    class Handle {
    public:
        Handle(BlockingPool<sqlite3*>::Handle pooled, sqlite3* fallback)
            : pooled_(std::move(pooled)), fallback_(fallback) {}

        sqlite3* get() const { return pooled_.valid() ? pooled_.value() : fallback_; }

    private:
        BlockingPool<sqlite3*>::Handle pooled_;  // invalid for an inert pool
        sqlite3* fallback_;
    };

    // Blocks until a connection is free (bounded by `size` concurrent
    // holders). Inert pools never block.
    Handle Acquire();

private:
    std::vector<std::unique_ptr<SqliteConnection>> connections_;  // owned; empty when inert
    std::optional<BlockingPool<sqlite3*>> pool_;                  // engaged only when not inert
    sqlite3* fallback_connection_;
};

}  // namespace retrieval_engine::detail
