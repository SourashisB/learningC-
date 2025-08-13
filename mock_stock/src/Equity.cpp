#include "Equity.h"
#include <chrono>

Equity::Equity(const std::string &symbol, double initPrice, double volatility, int initVolume)
    : symbol(symbol), price(initPrice), volatility(volatility), volume(initVolume),
      rng(std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count())),
      dist(0.0, 1.0) {}

void Equity::updatePrice() {
    double pctChange = dist(rng) * volatility;
    price *= (1.0 + pctChange);
    if (price < 0.01) price = 0.01;
    volume += static_cast<int>(dist(rng) * 1000);
    if (volume < 0) volume = 0;
}

void Equity::setPrice(double newPrice) {
    price = newPrice;
}

void Equity::setVolume(int newVolume) {
    volume = newVolume;
}

std::string Equity::getSymbol() const { return symbol; }
double Equity::getPrice() const { return price; }
double Equity::getVolatility() const { return volatility; }
int Equity::getVolume() const { return volume; }