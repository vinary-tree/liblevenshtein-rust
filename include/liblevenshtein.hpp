#ifndef LIBLEVENSHTEIN_HPP
#define LIBLEVENSHTEIN_HPP

#include "liblevenshtein.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vinary_tree::liblevenshtein {

class error final : public std::runtime_error {
public:
    explicit error(LlevStatus status)
        : std::runtime_error(llev_last_error_message()), status_(status) {}

    [[nodiscard]] LlevStatus status() const noexcept { return status_; }

private:
    LlevStatus status_;
};

inline void check(LlevStatus status) {
    if (status != LLEV_STATUS_OK) {
        throw error(status);
    }
}

enum class algorithm : std::uint32_t {
    standard = LLEV_ALGORITHM_STANDARD,
    transposition = LLEV_ALGORITHM_TRANSPOSITION,
    merge_and_split = LLEV_ALGORITHM_MERGE_AND_SPLIT,
    damerau_levenshtein = LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN,
};

enum class query_order : std::uint32_t {
    traversal = LLEV_QUERY_ORDER_TRAVERSAL,
    distance_then_term = LLEV_QUERY_ORDER_DISTANCE_THEN_TERM,
};

namespace detail {

struct cursor_state final {
    explicit cursor_state(LlevQueryCursor* cursor) noexcept : value(cursor) {}
    cursor_state(const cursor_state&) = delete;
    cursor_state& operator=(const cursor_state&) = delete;
    ~cursor_state() {
        if (value != nullptr) {
            (void)llev_query_cursor_free(value);
        }
    }

    LlevQueryCursor* value;
};

} // namespace detail

class batch final {
public:
    batch() = default;
    batch(std::shared_ptr<detail::cursor_state> cursor,
          LlevMatchBatchView view) noexcept
        : cursor_(std::move(cursor)), view_(view) {}
    batch(const batch&) = delete;
    batch& operator=(const batch&) = delete;
    batch(batch&& other) noexcept
        : cursor_(std::move(other.cursor_)), view_(other.view_) {}
    batch& operator=(batch&& other) noexcept {
        if (this != &other) {
            release();
            cursor_ = std::move(other.cursor_);
            view_ = other.view_;
        }
        return *this;
    }
    ~batch() { release(); }

    [[nodiscard]] std::span<const LlevMatch> matches() const noexcept {
        return {view_.matches, view_.len};
    }

    [[nodiscard]] static std::span<const std::byte> bytes(const LlevMatch& item) {
        if (item.unit_domain == VT_UNIT_DOMAIN_U64) {
            throw std::invalid_argument("u64 match is not a byte term");
        }
        return {static_cast<const std::byte*>(item.term_data), item.byte_len};
    }

    [[nodiscard]] static std::string_view utf8(const LlevMatch& item) {
        if (item.unit_domain != VT_UNIT_DOMAIN_UNICODE_SCALAR) {
            throw std::invalid_argument("match is not Unicode text");
        }
        return {static_cast<const char*>(item.term_data), item.byte_len};
    }

    [[nodiscard]] static std::span<const std::uint64_t> tokens(const LlevMatch& item) {
        if (item.unit_domain != VT_UNIT_DOMAIN_U64) {
            throw std::invalid_argument("match is not a u64 term");
        }
        return {static_cast<const std::uint64_t*>(item.term_data), item.term_len};
    }

private:
    void release() noexcept {
        if (cursor_) {
            (void)llev_query_cursor_release_batch(cursor_->value, view_.generation);
            cursor_.reset();
        }
    }

    std::shared_ptr<detail::cursor_state> cursor_;
    LlevMatchBatchView view_{};
};

class query_cursor final {
public:
    explicit query_cursor(LlevQueryCursor* value) try
        : state_(std::make_shared<detail::cursor_state>(value)) {}
    catch (...) {
        (void)llev_query_cursor_free(value);
        throw;
    }
    query_cursor(const query_cursor&) = delete;
    query_cursor& operator=(const query_cursor&) = delete;
    query_cursor(query_cursor&&) noexcept = default;
    query_cursor& operator=(query_cursor&&) noexcept = default;
    ~query_cursor() = default;

    [[nodiscard]] batch next_batch(std::size_t maximum = LLEV_DEFAULT_MATCH_BATCH) {
        LlevMatchBatchView view{};
        const auto status = llev_query_cursor_next_batch(state_->value, maximum, &view);
        if (status == LLEV_STATUS_END) {
            return {};
        }
        check(status);
        return batch(state_, view);
    }

private:
    std::shared_ptr<detail::cursor_state> state_;
};

class transducer final {
public:
    transducer(const VtResource& dictionary,
               algorithm selected = algorithm::standard) {
        check(llev_transducer_new(&dictionary,
                                  static_cast<std::uint32_t>(selected),
                                  &value_));
    }
    transducer(const transducer&) = delete;
    transducer& operator=(const transducer&) = delete;
    transducer(transducer&& other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    transducer& operator=(transducer&& other) noexcept {
        if (this != &other) {
            llev_transducer_free(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    ~transducer() { llev_transducer_free(value_); }

    [[nodiscard]] query_cursor query(
        std::string_view text,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_utf8(value_, text.data(), text.size(),
                                         maximum_distance,
                                         static_cast<std::uint32_t>(order),
                                         &cursor));
        return query_cursor(cursor);
    }

    [[nodiscard]] query_cursor query(
        std::span<const std::byte> bytes,
        std::size_t maximum_distance) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_bytes(
            value_, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(),
            maximum_distance, LLEV_QUERY_ORDER_TRAVERSAL, &cursor));
        return query_cursor(cursor);
    }

    [[nodiscard]] query_cursor query(
        std::span<const std::uint64_t> tokens,
        std::size_t maximum_distance) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_u64(value_, tokens.data(), tokens.size(),
                                        maximum_distance,
                                        LLEV_QUERY_ORDER_TRAVERSAL,
                                        &cursor));
        return query_cursor(cursor);
    }

private:
    LlevTransducer* value_ = nullptr;
};

} // namespace vinary_tree::liblevenshtein

#endif /* LIBLEVENSHTEIN_HPP */
