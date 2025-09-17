#pragma once
#include <vector>

struct SimulationResult {
    double var95;
    double var99;
    double cvar95;
    double cvar99;
};

SimulationResult runMonteCarloCPU_MT(
    const std::vector<std::vector<double>>& historicalReturns,
    const std::vector<double>& weights,
    int numSimulations,
    int numThreads
);