#include "read_connection_pool.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <utility>

namespace retrieval_engine::detail {

namespace {

bool HasStablePath(const std::string& db_path) { return !db_path.empty() && db_path != ":memory:"; }

}  // namespace

ReadConnectionPool::ReadConnectionPool(const std::string& db_path, sqlite3* fallback_connection, std::size_t size)
    : fallback_connection_(fallback_connection) {
    if (!HasStablePath(db_path)) return;  // inert: pool_ stays disengaged

    const std::size_t pool_size = std::max<std::size_t>(1, size);
    std::vector<sqlite3*> handles;
    connections_.reserve(pool_size);
    handles.reserve(pool_size);
    for (std::size_t i = 0; i < pool_size; ++i) {
        connections_.push_back(std::make_unique<SqliteConnection>(db_path, SQLITE_OPEN_READONLY));
        handles.push_back(connections_.back()->get());
    }
    pool_.emplace(std::move(handles));
}

ReadConnectionPool::Handle ReadConnectionPool::Acquire() {
    if (!pool_) return Handle({}, fallback_connection_);
    return Handle(pool_->Acquire(), nullptr);
}

}  // namespace retrieval_engine::detail
