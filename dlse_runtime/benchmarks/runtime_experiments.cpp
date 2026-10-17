#include "dlse/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct RequestSpec { std::size_t prompt{}; std::size_t output{}; };
struct Workload { std::string name; std::vector<RequestSpec> requests; };
struct SchedulerResult {
    std::string workload;
    std::size_t batch{}, budget{}, chunk{}, iterations{};
    std::size_t prompt_tokens{}, decode_tokens{}, max_decode_gap{};
    double p50_gap{}, p95_gap{}, p99_gap{}, mean_prefill_tokens{};
    std::size_t max_prefill_chunk{};
};
struct RequestState {
    dlse::RequestId id{};
    std::size_t prompt{}, output{}, prefilled{}, generated{};
    std::vector<std::uint64_t> decode_iterations;
};

std::vector<Workload> make_workloads(std::size_t batch) {
    Workload decode{"decode_heavy", {}};
    for (std::size_t i = 0; i < batch + 4; ++i) decode.requests.push_back({0, 128});

    Workload mixed{"mixed_long_prompt", {}};
    for (std::size_t i = 0; i < batch + 4; ++i) mixed.requests.push_back({0, 128});
    for (const auto prompt : {4096U, 2049U, 1023U, 257U}) mixed.requests.push_back({prompt, 32});

    Workload burst{"prompt_burst", {}};
    for (const auto prompt : {257U, 1024U, 2049U, 4096U, 513U, 128U, 777U, 1537U})
        burst.requests.push_back({prompt, 32});

    return {std::move(decode), std::move(mixed), std::move(burst)};
}

double percentile(const std::vector<std::size_t>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(sorted.size()))) - 1;
    return static_cast<double>(sorted[std::min(index, sorted.size() - 1)]);
}

SchedulerResult run_scheduler_case(
    const Workload& workload, std::size_t batch,
    std::size_t budget, std::size_t chunk) {
    dlse::ContinuousBatchScheduler scheduler({batch, budget, chunk});
    std::vector<RequestState> requests;
    std::unordered_map<dlse::RequestId, std::size_t> index;
    for (const auto& spec : workload.requests) {
        const auto id = scheduler.submit(spec.prompt, spec.output);
        index.emplace(id, requests.size());
        requests.push_back({id, spec.prompt, spec.output, 0, 0, {}});
    }

    std::size_t prompt_tokens = 0, decode_tokens = 0;
    std::size_t max_gap = 0, max_chunk = 0, prefill_tokens_total = 0;
    std::vector<std::size_t> gaps;
    std::size_t iterations = 0;

    while (scheduler.snapshot().finished < requests.size()) {
        if (++iterations > 20000)
            throw std::runtime_error("scheduler experiment exceeded 20000 iterations");

        const auto plan = scheduler.dispatch();
        std::size_t step_tokens = 0;
        std::unordered_set<dlse::RequestId> decoded;

        for (const auto& item : plan.prefill) {
            auto& request = requests.at(index.at(item.request_id));
            if (item.token_offset != request.prefilled || item.token_count == 0 ||
                item.token_count > chunk ||
                item.token_offset + item.token_count > request.prompt)
                throw std::runtime_error("invalid prefill span in scheduler experiment");

            request.prefilled += item.token_count;
            prompt_tokens += item.token_count;
            prefill_tokens_total += item.token_count;
            step_tokens += item.token_count;
            max_chunk = std::max(max_chunk, item.token_count);
        }

        for (const auto& item : plan.decode) {
            auto& request = requests.at(index.at(item.request_id));
            if (!decoded.insert(item.request_id).second ||
                request.prefilled != request.prompt ||
                item.token_offset != request.generated || item.token_count != 1 ||
                request.generated >= request.output)
                throw std::runtime_error("invalid decode step in scheduler experiment");

            ++request.generated;
            request.decode_iterations.push_back(plan.iteration);
            ++decode_tokens;
            ++step_tokens;
        }

        if (step_tokens > budget || plan.decode.size() > batch)
            throw std::runtime_error("scheduler configuration limit was exceeded");
    }

    for (const auto& request : requests) {
        if (request.prefilled != request.prompt || request.generated != request.output)
            throw std::runtime_error("scheduler experiment ended with an incomplete request");
        for (std::size_t i = 1; i < request.decode_iterations.size(); ++i) {
            const auto gap = static_cast<std::size_t>(
                request.decode_iterations[i] - request.decode_iterations[i - 1]);
            gaps.push_back(gap);
            max_gap = std::max(max_gap, gap);
        }
    }

    std::sort(gaps.begin(), gaps.end());
    const double mean_prefill = iterations == 0 ? 0.0 :
        static_cast<double>(prefill_tokens_total) / static_cast<double>(iterations);
    return {workload.name, batch, budget, chunk, iterations, prompt_tokens,
        decode_tokens, max_gap, percentile(gaps, 0.50), percentile(gaps, 0.95),
        percentile(gaps, 0.99), mean_prefill, max_chunk};
}

struct KvResult {
    std::string workload;
    std::size_t page_size{}, sequences{}, logical_tokens{}, reserved_tokens{};
    std::size_t reserved_bytes{}, contiguous_bytes{};
    double reduction{}, fragmentation{};
};

std::vector<KvResult> run_kv_sweep() {
    constexpr std::size_t max_seq_len = 4096, batch = 8, bytes_per_token = 4096;
    const std::array<std::pair<std::string, std::vector<std::size_t>>, 4> workloads{{
        {"short", {32, 64, 96, 128, 64, 48, 80, 40}},
        {"mixed", {128, 256, 512, 768, 1024, 128, 64, 512}},
        {"ragged", {127, 255, 513, 767, 1021, 130, 65, 509}},
        {"long", {3072, 4096, 2048, 3840, 1024, 1024, 2048, 3584}}
    }};

    std::vector<KvResult> results;
    for (const auto page_size : {8U, 16U, 32U, 64U}) {
        const auto max_pages = batch * (max_seq_len / page_size);
        for (const auto& [name, lengths] : workloads) {
            dlse::PagedKVCache cache({page_size, max_pages, bytes_per_token});
            for (std::size_t i = 0; i < lengths.size(); ++i) {
                const auto id = static_cast<dlse::RequestId>(i + 1);
                cache.create(id);
                cache.append(id, lengths[i]);
            }
            const auto stats = cache.stats();
            const std::size_t contiguous = batch * max_seq_len * bytes_per_token;
            const double reduction = 1.0 -
                static_cast<double>(stats.reserved_bytes) /
                static_cast<double>(contiguous);
            results.push_back({name, page_size, lengths.size(), stats.logical_tokens,
                stats.reserved_tokens, stats.reserved_bytes, contiguous, reduction,
                stats.internal_fragmentation});
        }
    }
    return results;
}

void write_scheduler_csv(const std::filesystem::path& path,
                         const std::vector<SchedulerResult>& results) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "workload,batch_size,token_budget,prefill_chunk,iterations,prompt_tokens,"
           "decode_tokens,max_decode_gap_iterations,p50_decode_gap_iterations,"
           "p95_decode_gap_iterations,p99_decode_gap_iterations,"
           "mean_prefill_tokens_per_iteration,max_prefill_chunk\n";
    out << std::fixed << std::setprecision(3);
    for (const auto& r : results) {
        out << r.workload << ',' << r.batch << ',' << r.budget << ',' << r.chunk << ','
            << r.iterations << ',' << r.prompt_tokens << ',' << r.decode_tokens << ','
            << r.max_decode_gap << ',' << r.p50_gap << ',' << r.p95_gap << ','
            << r.p99_gap << ',' << r.mean_prefill_tokens << ','
            << r.max_prefill_chunk << '\n';
    }
}

void write_kv_csv(const std::filesystem::path& path,
                  const std::vector<KvResult>& results) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "workload,page_size_tokens,sequences,logical_tokens,reserved_tokens,"
           "reserved_kv_bytes,contiguous_kv_bytes,reservation_reduction_fraction,"
           "internal_fragmentation\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        out << r.workload << ',' << r.page_size << ',' << r.sequences << ','
            << r.logical_tokens << ',' << r.reserved_tokens << ','
            << r.reserved_bytes << ',' << r.contiguous_bytes << ','
            << r.reduction << ',' << r.fragmentation << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output_dir = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("dlse_runtime/results/experiments");
        if (argc > 2) {
            std::cerr << "usage: dlse_runtime_experiments [output-dir]\n";
            return 2;
        }
        std::filesystem::create_directories(output_dir);
        const std::array<std::size_t, 3> batches{4, 8, 16};
        const std::array<std::size_t, 3> budgets{32, 64, 128};
        const std::array<std::size_t, 4> chunks{8, 16, 32, 64};

        std::vector<SchedulerResult> scheduler_results;
        for (const auto batch : batches)
            for (const auto budget : budgets)
                for (const auto chunk : chunks)
                    for (const auto& workload : make_workloads(batch))
                        scheduler_results.push_back(
                            run_scheduler_case(workload, batch, budget, chunk));

        const auto kv_results = run_kv_sweep();
        const auto scheduler_path = output_dir / "scheduler_sweep.csv";
        const auto kv_path = output_dir / "kv_cache_sweep.csv";
        write_scheduler_csv(scheduler_path, scheduler_results);
        write_kv_csv(kv_path, kv_results);

        std::cout << "scheduler_parameter_sets=36\n"
                  << "scheduler_workloads=3\n"
                  << "scheduler_records=" << scheduler_results.size() << '\n'
                  << "kv_page_sizes=4\n"
                  << "kv_workloads=4\n"
                  << "kv_records=" << kv_results.size() << '\n'
                  << "scheduler_csv=" << scheduler_path.string() << '\n'
                  << "kv_csv=" << kv_path.string() << '\n';

        for (const auto& r : scheduler_results) {
            std::cout << "SCHED_ROW," << r.workload << ',' << r.batch << ','
                      << r.budget << ',' << r.chunk << ',' << r.iterations << ','
                      << r.prompt_tokens << ',' << r.decode_tokens << ','
                      << r.max_decode_gap << ',' << std::fixed << std::setprecision(3)
                      << r.p50_gap << ',' << r.p95_gap << ',' << r.p99_gap << ','
                      << r.mean_prefill_tokens << ',' << r.max_prefill_chunk << '\n';
        }
        for (const auto& r : kv_results) {
            std::cout << "KV_ROW," << r.workload << ',' << r.page_size << ','
                      << r.sequences << ',' << r.logical_tokens << ','
                      << r.reserved_tokens << ',' << r.reserved_bytes << ','
                      << r.contiguous_bytes << ',' << std::fixed << std::setprecision(6)
                      << r.reduction << ',' << r.fragmentation << '\n';
        }
        std::cout << "experiment_status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "experiment_status=FAIL error=" << error.what() << '\n';
        return 1;
    }
}
