#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"
#include "montecarlo.hpp"

using json = nlohmann::json;

std::vector<std::vector<double>> loadHistoricalReturns(const std::string& csvPath) {
    std::vector<std::vector<double>> data;
    std::ifstream file(csvPath);
    if (!file.is_open()) throw std::runtime_error("Could not open CSV file");

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::vector<double> row;
        size_t pos = 0;
        int colIndex = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            std::string token = line.substr(0, pos);
            if (colIndex >= 1) { // skip Date col
                row.push_back(std::stod(token));
            }
            line.erase(0, pos + 1);
            colIndex++;
        }
        row.push_back(std::stod(line)); // last col
        data.push_back(row);
    }
    return data;
}

std::vector<double> loadAllocations(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) throw std::runtime_error("Could not open JSON file");

    json j;
    file >> j;
    std::vector<double> w;
    for (auto& el : j.items()) {
        w.push_back(el.value());
    }
    return w;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: risk_engine <allocations.json> <returns.csv> <output.json> <threads>\n";
        return 1;
    }

    try {
        auto weights = loadAllocations(argv[1]);
        auto histReturns = loadHistoricalReturns(argv[2]);
        int threads = std::stoi(argv[4]);

        int numSimulations = 5'000'000; // adjust as needed
        auto result = runMonteCarloCPU_MT(histReturns, weights, numSimulations, threads);

        json out;
        out["VaR_95"]  = result.var95;
        out["VaR_99"]  = result.var99;
        out["CVaR_95"] = result.cvar95;
        out["CVaR_99"] = result.cvar99;

        std::ofstream outFile(argv[3]);
        outFile << out.dump(4);

        std::cout << "Risk metrics saved to " << argv[3] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}