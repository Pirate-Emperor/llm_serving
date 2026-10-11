#include "dlse/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void require(
    bool ok,
    const char* expression,
    std::source_location loc = std::source_location::current()) {
    if (!ok) {
        throw std::runtime_error(
            std::string(loc.file_name()) + ":" +
            std::to_string(loc.line()) +
            ": check failed: " + expression);
    }
}

#define CHECK(expr) require(static_cast<bool>(expr), #expr)

template <class Exception, class Fn>
void expect_throw(Fn&& fn) {
    bool caught = false;
    try {
        std::forward<Fn>(fn)();
    } catch (const Exception&) {
        caught = true;
    }
    CHECK(caught);
}

void scheduler_invalid_config() {
    expect_throw<std::invalid_argument>([] {
        dlse::ContinuousBatchScheduler scheduler({0, 16, 4});
    });
    expect_throw<std::invalid_argument>([] {
        dlse::ContinuousBatchScheduler scheduler({4, 0, 4});
    });
    expect_throw<std::invalid_argument>([] {
        dlse::ContinuousBatchScheduler scheduler({4, 16, 0});
    });
    expect_throw<std::invalid_argument>([] {
        dlse::ContinuousBatchScheduler scheduler({4, 3, 1});
    });
}

void scheduler_zero_generation_rejected() {
    dlse::ContinuousBatchScheduler scheduler({4, 16, 4});
    expect_throw<std::invalid_argument>([&] {
        (void)scheduler.submit(12, 0);
    });
    CHECK(scheduler.snapshot().submitted == 0);
}

void scheduler_unknown_request_lookup() {
    dlse::ContinuousBatchScheduler scheduler({4, 16, 4});
    CHECK(!scheduler.get(12345).has_value());
}

void scheduler_chunk_offsets_and_transition() {
    dlse::ContinuousBatchScheduler scheduler({4, 16, 4});
    const auto id = scheduler.submit(10, 3);

    CHECK(scheduler.snapshot().waiting == 1);
    CHECK(scheduler.snapshot().active == 0);

    const auto a = scheduler.dispatch();
    CHECK(a.prefill.size() == 1);
    CHECK(a.prefill[0].request_id == id);
    CHECK(a.prefill[0].token_offset == 0);
    CHECK(a.prefill[0].token_count == 4);
    CHECK(scheduler.get(id)->prefill_cursor == 4);

    const auto b = scheduler.dispatch();
    CHECK(b.prefill.size() == 1);
    CHECK(b.prefill[0].token_offset == 4);
    CHECK(b.prefill[0].token_count == 4);
    CHECK(scheduler.get(id)->prefill_cursor == 8);

    const auto c = scheduler.dispatch();
    CHECK(c.prefill.size() == 1);
    CHECK(c.prefill[0].token_offset == 8);
    CHECK(c.prefill[0].token_count == 2);
    CHECK(scheduler.get(id)->prefill_cursor == 10);
    CHECK(scheduler.snapshot().active == 1);

    const auto d = scheduler.dispatch();
    CHECK(d.decode.size() == 1);
    CHECK(d.decode[0].request_id == id);
    CHECK(d.decode[0].token_offset == 0);
    CHECK(d.decode[0].token_count == 1);
}

void scheduler_zero_prompt_starts_decode() {
    dlse::ContinuousBatchScheduler scheduler({2, 8, 4});
    const auto id = scheduler.submit(0, 2);
    CHECK(scheduler.snapshot().waiting == 0);
    CHECK(scheduler.snapshot().active == 1);

    const auto first = scheduler.dispatch();
    CHECK(first.decode.size() == 1);
    CHECK(first.decode[0].request_id == id);
    CHECK(first.decode[0].token_offset == 0);

    const auto second = scheduler.dispatch();
    CHECK(second.decode.size() == 1);
    CHECK(second.decode[0].token_offset == 1);
    CHECK(scheduler.get(id)->finished);
    CHECK(scheduler.snapshot().finished == 1);
    CHECK(scheduler.snapshot().active == 0);
}

void scheduler_decode_is_prioritized() {
    dlse::ContinuousBatchScheduler scheduler({4, 16, 4});
    for (int i = 0; i < 4; ++i) {
        (void)scheduler.submit(0, 100);
    }
    (void)scheduler.dispatch();

    const auto prompt = scheduler.submit(4096, 8);
    const auto plan = scheduler.dispatch();
    CHECK(plan.decode.size() == 4);
    CHECK(plan.prefill.size() == 1);
    CHECK(plan.prefill[0].request_id == prompt);
    CHECK(plan.prefill[0].token_count <= 4);
    CHECK(plan.decode.size() + plan.prefill[0].token_count <= 16);
}

void scheduler_decode_batch_limit() {
    dlse::ContinuousBatchScheduler scheduler({4, 32, 8});
    for (int i = 0; i < 11; ++i) {
        (void)scheduler.submit(0, 20);
    }
    const auto plan = scheduler.dispatch();
    CHECK(plan.decode.size() == 4);
}

void scheduler_total_token_budget() {
    dlse::ContinuousBatchScheduler scheduler({2, 4, 3});
    const auto prompt = scheduler.submit(10, 10);
    (void)scheduler.submit(0, 10);
    const auto plan = scheduler.dispatch();

    std::size_t tokens = 0;
    for (const auto& item : plan.decode) {
        tokens += item.token_count;
    }
    for (const auto& item : plan.prefill) {
        tokens += item.token_count;
    }

    CHECK(tokens <= 4);
    CHECK(plan.decode.size() == 1);
    CHECK(plan.prefill.size() == 1);
    CHECK(plan.prefill[0].request_id == prompt);
    CHECK(plan.prefill[0].token_count == 3);
}

void scheduler_prefill_round_robin() {
    dlse::ContinuousBatchScheduler scheduler({2, 8, 4});
    const auto first_id = scheduler.submit(12, 2);
    const auto second_id = scheduler.submit(8, 2);

    const auto first = scheduler.dispatch();
    CHECK(first.prefill.size() == 2);
    CHECK(first.prefill[0].request_id == first_id);
    CHECK(first.prefill[1].request_id == second_id);
    CHECK(first.prefill[0].token_count == 4);
    CHECK(first.prefill[1].token_count == 4);

    const auto second = scheduler.dispatch();
    CHECK(second.prefill.size() == 2);
    CHECK(second.prefill[0].request_id == first_id);
    CHECK(second.prefill[0].token_offset == 4);
    CHECK(second.prefill[1].request_id == second_id);
    CHECK(second.prefill[1].token_offset == 4);
}

void scheduler_one_chunk_per_waiting_request() {
    dlse::ContinuousBatchScheduler scheduler({8, 64, 8});
    const auto id = scheduler.submit(100, 2);
    const auto plan = scheduler.dispatch();
    CHECK(plan.prefill.size() == 1);
    CHECK(plan.prefill[0].request_id == id);
    CHECK(plan.prefill[0].token_count == 8);
}

void scheduler_max_new_tokens_finishes() {
    dlse::ContinuousBatchScheduler scheduler({2, 8, 4});
    const auto id = scheduler.submit(0, 2);

    CHECK(scheduler.dispatch().decode.size() == 1);
    CHECK(!scheduler.get(id)->finished);
    CHECK(scheduler.dispatch().decode.size() == 1);
    CHECK(scheduler.get(id)->finished);
    CHECK(scheduler.snapshot().finished == 1);
    CHECK(scheduler.dispatch().decode.empty());
}

void scheduler_finish_is_idempotent() {
    dlse::ContinuousBatchScheduler scheduler({2, 8, 4});
    const auto id = scheduler.submit(32, 4);

    scheduler.finish(id);
    scheduler.finish(id);
    scheduler.abort(id);

    const auto snap = scheduler.snapshot();
    CHECK(snap.waiting == 0);
    CHECK(snap.active == 0);
    CHECK(snap.finished == 1);
    CHECK(snap.submitted == 1);
    CHECK(scheduler.dispatch().prefill.empty());
}

void scheduler_abort_is_idempotent() {
    dlse::ContinuousBatchScheduler scheduler({2, 8, 4});
    const auto id = scheduler.submit(0, 100);
    (void)scheduler.dispatch();

    scheduler.abort(id);
    scheduler.abort(id);
    scheduler.finish(id);

    CHECK(scheduler.get(id)->finished);
    CHECK(scheduler.snapshot().active == 0);
    CHECK(scheduler.snapshot().finished == 1);
    CHECK(scheduler.dispatch().decode.empty());
}

void scheduler_snapshot_tracks_live_queues() {
    dlse::ContinuousBatchScheduler scheduler({4, 16, 4});
    const auto waiting_id = scheduler.submit(10, 5);
    const auto active_id = scheduler.submit(0, 5);

    auto snap = scheduler.snapshot();
    CHECK(snap.waiting == 1);
    CHECK(snap.active == 1);
    CHECK(snap.submitted == 2);
    CHECK(snap.finished == 0);

    scheduler.abort(waiting_id);
    snap = scheduler.snapshot();
    CHECK(snap.waiting == 0);
    CHECK(snap.active == 1);
    CHECK(snap.finished == 1);

    scheduler.finish(active_id);
    snap = scheduler.snapshot();
    CHECK(snap.waiting == 0);
    CHECK(snap.active == 0);
    CHECK(snap.finished == 2);
}

void kv_invalid_configs() {
    expect_throw<std::invalid_argument>([] {
        dlse::PagedKVCache cache({0, 8, 1});
    });
    expect_throw<std::invalid_argument>([] {
        dlse::PagedKVCache cache({16, 0, 1});
    });
    if (sizeof(std::size_t) > sizeof(dlse::PhysicalPage)) {
        const auto too_many =
            static_cast<std::size_t>(
                std::numeric_limits<dlse::PhysicalPage>::max()) + 1;
        expect_throw<std::invalid_argument>([&] {
            dlse::PagedKVCache cache({16, too_many, 1});
        });
    }
}

void kv_empty_and_zero_append() {
    dlse::PagedKVCache cache({16, 4, 32});
    cache.create(1);
    cache.append(1, 0);
    CHECK(cache.logical_tokens(1) == 0);
    CHECK(cache.block_table(1).empty());
    const auto stats = cache.stats();
    CHECK(stats.logical_tokens == 0);
    CHECK(stats.reserved_tokens == 0);
    CHECK(stats.reserved_bytes == 0);
    CHECK(stats.internal_fragmentation == 0.0);
}

void kv_exact_page_boundaries() {
    dlse::PagedKVCache cache({16, 8, 2});
    cache.create(1);
    cache.append(1, 15);
    CHECK(cache.block_table(1).size() == 1);
    cache.append(1, 1);
    CHECK(cache.block_table(1).size() == 1);
    cache.append(1, 1);
    CHECK(cache.block_table(1).size() == 2);
    CHECK(cache.logical_tokens(1) == 17);
    CHECK(cache.stats().reserved_tokens == 32);
}

void kv_pages_do_not_alias_between_sequences() {
    dlse::PagedKVCache cache({16, 32, 8});
    for (std::uint64_t id = 1; id <= 12; ++id) {
        cache.create(id);
        cache.append(id, static_cast<std::size_t>(id));
    }

    std::set<dlse::PhysicalPage> pages;
    for (std::uint64_t id = 1; id <= 12; ++id) {
        for (const auto page : cache.block_table(id)) {
            CHECK(pages.insert(page).second);
        }
    }
    CHECK(cache.stats().used_pages == pages.size());
}

void kv_stats_bytes_and_fragmentation() {
    dlse::PagedKVCache cache({16, 8, 32});
    cache.create(5);
    cache.append(5, 17);

    const auto stats = cache.stats();
    CHECK(stats.total_pages == 8);
    CHECK(stats.free_pages == 6);
    CHECK(stats.used_pages == 2);
    CHECK(stats.logical_tokens == 17);
    CHECK(stats.reserved_tokens == 32);
    CHECK(stats.reserved_bytes == 1024);
    CHECK(std::abs(stats.internal_fragmentation - 15.0 / 32.0) < 1e-12);
}

void kv_release_reclaims_capacity() {
    dlse::PagedKVCache cache({16, 4, 16});
    cache.create(9);
    cache.append(9, 33);
    CHECK(cache.stats().used_pages == 3);

    cache.release(9);
    const auto stats = cache.stats();
    CHECK(stats.used_pages == 0);
    CHECK(stats.free_pages == 4);
    CHECK(stats.logical_tokens == 0);
    CHECK(stats.reserved_bytes == 0);
    expect_throw<std::out_of_range>([&] {
        (void)cache.block_table(9);
    });
}

void kv_release_unknown_is_noop() {
    dlse::PagedKVCache cache({16, 4, 16});
    cache.create(1);
    cache.append(1, 17);
    cache.release(987);
    CHECK(cache.stats().used_pages == 2);
    cache.release(1);
    cache.release(1);
    CHECK(cache.stats().used_pages == 0);
}

void kv_duplicate_create_rejected() {
    dlse::PagedKVCache cache({16, 4, 1});
    cache.create(3);
    expect_throw<std::invalid_argument>([&] {
        cache.create(3);
    });
}

void kv_unknown_requests_rejected() {
    dlse::PagedKVCache cache({16, 4, 1});
    expect_throw<std::out_of_range>([&] {
        cache.append(8, 1);
    });
    expect_throw<std::out_of_range>([&] {
        (void)cache.block_table(8);
    });
    expect_throw<std::out_of_range>([&] {
        (void)cache.logical_tokens(8);
    });
}

void kv_exhaustion_is_atomic() {
    dlse::PagedKVCache cache({16, 1, 8});
    cache.create(1);
    cache.create(2);
    cache.append(1, 16);

    expect_throw<std::runtime_error>([&] {
        cache.append(2, 1);
    });

    auto stats = cache.stats();
    CHECK(stats.used_pages == 1);
    CHECK(stats.free_pages == 0);
    CHECK(stats.logical_tokens == 16);
    CHECK(stats.reserved_tokens == 16);
    CHECK(cache.logical_tokens(2) == 0);
    CHECK(cache.block_table(2).empty());

    cache.release(1);
    cache.append(2, 1);
    CHECK(cache.logical_tokens(2) == 1);
    CHECK(cache.stats().used_pages == 1);
}

void kv_reuses_released_page_ids() {
    dlse::PagedKVCache cache({16, 2, 4});
    cache.create(1);
    cache.append(1, 16);
    const auto old_page = cache.block_table(1).front();
    cache.release(1);

    cache.create(2);
    cache.append(2, 16);
    CHECK(cache.block_table(2).front() == old_page);
}

void kv_page_count_handles_size_max() {
    dlse::PagedKVCache cache({16, 2, 1});
    const auto max_tokens = std::numeric_limits<std::size_t>::max();
    CHECK(cache.pages_for_tokens(max_tokens) ==
          1 + (max_tokens - 1) / 16);

    cache.create(1);
    expect_throw<std::runtime_error>([&] {
        cache.append(1, max_tokens);
    });
    CHECK(cache.logical_tokens(1) == 0);
    CHECK(cache.stats().used_pages == 0);

    cache.append(1, 1);
    CHECK(cache.logical_tokens(1) == 1);
}

void kv_logical_overflow_leaves_state_unchanged() {
    dlse::PagedKVCache cache({16, 4, 1});
    cache.create(1);
    cache.append(1, 1);
    expect_throw<std::overflow_error>([&] {
        cache.append(1, std::numeric_limits<std::size_t>::max());
    });
    CHECK(cache.logical_tokens(1) == 1);
    CHECK(cache.stats().used_pages == 1);
}

void randomized_scheduler_invariants() {
    struct ModelRequest {
        dlse::RequestId id{};
        std::size_t prompt{};
        std::size_t max_new{};
        std::size_t prefilled{};
        std::size_t generated{};
    };

    constexpr std::uint64_t seed = 0xD15EA5EULL;
    std::mt19937_64 random(seed);
    dlse::ContinuousBatchScheduler scheduler({8, 64, 16});
    std::vector<ModelRequest> requests;
    std::unordered_map<dlse::RequestId, std::size_t> index;

    for (int i = 0; i < 64; ++i) {
        const std::size_t prompt =
            static_cast<std::size_t>(random() % 257);
        const std::size_t max_new =
            1 + static_cast<std::size_t>(random() % 12);
        const auto id = scheduler.submit(prompt, max_new);
        index.emplace(id, requests.size());
        requests.push_back({id, prompt, max_new, 0, 0});
    }

    std::size_t total_iterations = 0;
    while (scheduler.snapshot().finished < requests.size()) {
        CHECK(++total_iterations <= 5000);
        const auto plan = scheduler.dispatch();
        CHECK(plan.iteration > 0);
        CHECK(plan.decode.size() <= 8);

        std::size_t tokens = 0;
        std::unordered_set<dlse::RequestId> decoded_this_iteration;

        for (const auto& item : plan.prefill) {
            const auto pos = index.at(item.request_id);
            auto& request = requests[pos];
            CHECK(item.kind == dlse::WorkKind::Prefill);
            CHECK(item.token_offset == request.prefilled);
            CHECK(item.token_count > 0);
            CHECK(item.token_count <= 16);
            CHECK(item.token_offset + item.token_count <= request.prompt);
            request.prefilled += item.token_count;
            tokens += item.token_count;
        }

        for (const auto& item : plan.decode) {
            const auto pos = index.at(item.request_id);
            auto& request = requests[pos];
            CHECK(decoded_this_iteration.insert(item.request_id).second);
            CHECK(item.kind == dlse::WorkKind::Decode);
            CHECK(request.prefilled == request.prompt);
            CHECK(item.token_count == 1);
            CHECK(item.token_offset == request.generated);
            CHECK(request.generated < request.max_new);
            ++request.generated;
            tokens += item.token_count;
        }

        CHECK(tokens <= 64);
    }

    CHECK(scheduler.snapshot().submitted == requests.size());
    CHECK(scheduler.snapshot().finished == requests.size());
    for (const auto& request : requests) {
        CHECK(request.prefilled == request.prompt);
        CHECK(request.generated == request.max_new);
        CHECK(scheduler.get(request.id)->finished);
    }
}

void randomized_kv_allocator_invariants() {
    constexpr std::uint64_t seed = 0xCACE1234ULL;
    std::mt19937_64 random(seed);
    constexpr std::size_t page_size = 16;
    constexpr std::size_t bytes_per_token = 32;
    constexpr std::size_t count = 16;

    dlse::PagedKVCache cache({page_size, 1024, bytes_per_token});
    std::vector<dlse::RequestId> ids(count);
    std::vector<std::size_t> lengths(count, 0);

    for (std::size_t i = 0; i < count; ++i) {
        ids[i] = static_cast<dlse::RequestId>(1000 + i);
        cache.create(ids[i]);
    }

    for (std::size_t step = 0; step < 2000; ++step) {
        const std::size_t i =
            static_cast<std::size_t>(random() % count);

        if (random() % 13 == 0) {
            cache.release(ids[i]);
            cache.create(ids[i]);
            lengths[i] = 0;
        } else {
            const std::size_t add =
                std::min<std::size_t>(
                    static_cast<std::size_t>(random() % 48),
                    512 - lengths[i]);
            cache.append(ids[i], add);
            lengths[i] += add;
        }

        std::size_t logical = 0;
        std::size_t pages = 0;
        std::unordered_set<dlse::PhysicalPage> unique_pages;

        for (std::size_t j = 0; j < count; ++j) {
            CHECK(cache.logical_tokens(ids[j]) == lengths[j]);
            const auto& table = cache.block_table(ids[j]);
            const std::size_t expected_pages =
                lengths[j] == 0 ? 0 : 1 + (lengths[j] - 1) / page_size;
            CHECK(table.size() == expected_pages);
            logical += lengths[j];
            pages += expected_pages;
            for (const auto page : table) {
                CHECK(unique_pages.insert(page).second);
            }
        }

        const auto stats = cache.stats();
        CHECK(stats.logical_tokens == logical);
        CHECK(stats.used_pages == pages);
        CHECK(stats.free_pages + stats.used_pages == stats.total_pages);
        CHECK(stats.reserved_tokens == pages * page_size);
        CHECK(stats.reserved_bytes == pages * page_size * bytes_per_token);
        const double expected_fragmentation =
            stats.reserved_tokens == 0
                ? 0.0
                : 1.0 -
                    static_cast<double>(logical) /
                    static_cast<double>(stats.reserved_tokens);
        CHECK(std::abs(
                  stats.internal_fragmentation -
                  expected_fragmentation) < 1e-12);
    }
}

struct TestCase {
    const char* suite;
    const char* name;
    std::function<void()> function;
};

const std::vector<TestCase>& tests() {
    static const std::vector<TestCase> all{
        {"scheduler", "rejects invalid limits", scheduler_invalid_config},
        {"scheduler", "rejects zero generation length", scheduler_zero_generation_rejected},
        {"scheduler", "unknown request lookup", scheduler_unknown_request_lookup},
        {"scheduler", "prefill chunk offsets and decode transition", scheduler_chunk_offsets_and_transition},
        {"scheduler", "zero prompt starts decode", scheduler_zero_prompt_starts_decode},
        {"scheduler", "decode is scheduled before prefill", scheduler_decode_is_prioritized},
        {"scheduler", "decode batch size is bounded", scheduler_decode_batch_limit},
        {"scheduler", "total token budget is bounded", scheduler_total_token_budget},
        {"scheduler", "prefill rotation is round robin", scheduler_prefill_round_robin},
        {"scheduler", "one chunk per waiting request", scheduler_one_chunk_per_waiting_request},
        {"scheduler", "maximum output length finishes request", scheduler_max_new_tokens_finishes},
        {"scheduler", "finish is idempotent", scheduler_finish_is_idempotent},
        {"scheduler", "abort is idempotent", scheduler_abort_is_idempotent},
        {"scheduler", "snapshot tracks live queue counts", scheduler_snapshot_tracks_live_queues},
        {"kv", "rejects invalid capacity settings", kv_invalid_configs},
        {"kv", "zero append reserves nothing", kv_empty_and_zero_append},
        {"kv", "page boundary rounding", kv_exact_page_boundaries},
        {"kv", "physical pages do not alias", kv_pages_do_not_alias_between_sequences},
        {"kv", "reservation bytes and fragmentation", kv_stats_bytes_and_fragmentation},
        {"kv", "release reclaims capacity", kv_release_reclaims_capacity},
        {"kv", "unknown release is a no-op", kv_release_unknown_is_noop},
        {"kv", "duplicate sequence is rejected", kv_duplicate_create_rejected},
        {"kv", "unknown sequence operations are rejected", kv_unknown_requests_rejected},
        {"kv", "pool exhaustion does not mutate state", kv_exhaustion_is_atomic},
        {"kv", "released page IDs can be reused", kv_reuses_released_page_ids},
        {"kv", "page count handles size_t maximum", kv_page_count_handles_size_max},
        {"kv", "logical token overflow preserves state", kv_logical_overflow_leaves_state_unchanged},
        {"stress", "seeded randomized scheduler invariants", randomized_scheduler_invariants},
        {"stress", "seeded randomized KV allocator invariants", randomized_kv_allocator_invariants},
    };
    return all;
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view selected =
        argc > 1 ? std::string_view(argv[1]) : std::string_view("all");

    if (argc > 2 ||
        (selected != "all" && selected != "scheduler" &&
         selected != "kv" && selected != "stress" && selected != "--list")) {
        std::cerr << "usage: dlse_runtime_tests [all|scheduler|kv|stress|--list]\n";
        return 2;
    }

    if (selected == "--list") {
        for (const auto& test : tests()) {
            std::cout << test.suite << ": " << test.name << '\n';
        }
        return 0;
    }

    std::size_t passed = 0;
    std::size_t failed = 0;

    for (const auto& test : tests()) {
        if (selected != "all" && selected != test.suite) {
            continue;
        }

        try {
            test.function();
            ++passed;
            std::cout << "[PASS] " << test.suite << " / "
                      << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.suite << " / "
                      << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.suite << " / "
                      << test.name << ": unknown error\n";
        }
    }

    std::cout << "suite=" << selected
              << " passed=" << passed
              << " failed=" << failed << '\n';

    return failed == 0 && passed > 0 ? 0 : 1;
}
