#include "montecarlo.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <future>

static std::vector<double> simulateChunk(
    const std::vector<std::vector<double>>& returns,
    const std::vector<double>& weights,
    int numSimulations,
    unsigned int seed
) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, returns.size() - 1);

    std::vector<double> results;
    results.reserve(numSimulations);

    for (int i = 0; i < numSimulations; ++i) {
        int idx = dist(gen);
        double r = 0.0;
        for (size_t a = 0; a < weights.size(); ++a) {
            r += returns[idx][a] * weights[a];
        }
        results.push_back(r);
    }
    return results;
}

SimulationResult runMonteCarloCPU_MT(
    const std::vector<std::vector<double>>& historicalReturns,
    const std::vector<double>& weights,
    int numSimulations,
    int numThreads
) {
    if (historicalReturns.empty() || weights.empty()) {
        throw std::runtime_error("Empty input data for Monte Carlo simulation");
    }
    if (numThreads < 1) numThreads = 1;

    int simsPerThread = numSimulations / numThreads;
    std::vector<std::future<std::vector<double>>> futures;

    for (int t = 0; t < numThreads; ++t) {
        unsigned int seed = static_cast<unsigned int>(std::random_device{}()) + t;
        futures.push_back(std::async(std::launch::async, simulateChunk,
                                     std::cref(historicalReturns),
                                     std::cref(weights),
                                     simsPerThread,
                                     seed));
    }

    std::vector<double> allReturns;
    allReturns.reserve(numSimulations);

    for (auto& f : futures) {
        auto chunk = f.get();
        allReturns.insert(allReturns.end(), chunk.begin(), chunk.end());
    }

    std::sort(allReturns.begin(), allReturns.end());

    auto getVaR = [&](double alpha) {
        size_t pos = static_cast<size_t>(alpha * allReturns.size());
        return allReturns[pos];
    };

    auto getCVaR = [&](double alpha) {
        size_t pos = static_cast<size_t>(alpha * allReturns.size());
        double sum = std::accumulate(allReturns.begin(), allReturns.begin() + pos, 0.0);
        return sum / pos;
    };

    SimulationResult res;
    res.var95  = getVaR(0.05);
    res.var99  = getVaR(0.01);
    res.cvar95 = getCVaR(0.05);
    res.cvar99 = getCVaR(0.01);
    return res;
}