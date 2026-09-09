#include "read_connection_pool.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <utility>

namespace retrieval_engine::detail {

namespace {

bool IsPoolable(const std::string& db_path) { return !db_path.empty() && db_path != ":memory:"; }

}  // namespace

ReadConnectionPool::ReadConnectionPool(const std::string& db_path, sqlite3* fallback_connection, std::size_t size)
    : fallback_connection_(fallback_connection), inert_(!IsPoolable(db_path)) {
    if (inert_) return;

    const std::size_t pool_size = std::max<std::size_t>(1, size);
    connections_.reserve(pool_size);
    free_.reserve(pool_size);
    for (std::size_t i = 0; i < pool_size; ++i) {
        connections_.push_back(std::make_unique<SqliteConnection>(db_path, SQLITE_OPEN_READONLY));
        free_.push_back(connections_.back()->get());
    }
}

ReadConnectionPool::Handle::~Handle() {
    if (pool_) pool_->Return(connection_);
}

ReadConnectionPool::Handle ReadConnectionPool::Acquire() {
    if (inert_) return Handle(nullptr, fallback_connection_);

    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] { return !free_.empty(); });
    sqlite3* connection = free_.back();
    free_.pop_back();
    return Handle(this, connection);
}

void ReadConnectionPool::Return(sqlite3* connection) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(connection);
    }
    available_.notify_one();
}

}  // namespace retrieval_engine::detail
