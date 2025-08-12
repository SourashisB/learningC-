#ifndef EQUITY_H
#define EQUITY_H

#include <string>
#include <random>

class Equity {
private:
    std::string symbol;
    double price;
    double volatility;
    int volume;
    std::mt19937 rng;
    std::normal_distribution<double> dist;

public:
    Equity(const std::string &symbol, double initPrice, double volatility, int initVolume);

    void updatePrice();
    std::string getSymbol() const;
    double getPrice() const;
    double getVolatility() const;
    int getVolume() const;
};

#endif