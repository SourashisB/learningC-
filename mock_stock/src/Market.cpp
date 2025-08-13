#include "Market.h"
#include <sstream>
#include <ctime>
#include <random>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PatternState {
    std::string type; // "none", "uptrend", "downtrend", "spike", "crash", "oscillate"
    int remainingTicks = 0;
    double strength = 0.0;
};

static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uni(0.0, 1.0);

Market::Market() : tick(0) {}

void Market::addEquity(const Equity &eq) {
    equities.push_back(eq);
}

void Market::step() {
    static std::vector<PatternState> patterns(equities.size());

    // Randomly start patterns
    for (size_t i = 0; i < equities.size(); i++) {
        if (patterns[i].remainingTicks <= 0 && uni(rng) < 0.05) { // 5% chance
            double r = uni(rng);
            if (r < 0.2) patterns[i].type = "uptrend";
            else if (r < 0.4) patterns[i].type = "downtrend";
            else if (r < 0.6) patterns[i].type = "spike";
            else if (r < 0.8) patterns[i].type = "crash";
            else patterns[i].type = "oscillate";

            patterns[i].remainingTicks = 5 + (int)(uni(rng) * 10); // 5–15 ticks
            patterns[i].strength = 0.005 + uni(rng) * 0.02; // 0.5%–2% per tick
        }
    }

    // Apply patterns & correlation
    for (size_t i = 0; i < equities.size(); i++) {
        Equity &eq = equities[i];
        double factor = 1.0;

        if (patterns[i].remainingTicks > 0) {
            if (patterns[i].type == "uptrend") {
                factor += patterns[i].strength;
            } else if (patterns[i].type == "downtrend") {
                factor -= patterns[i].strength;
            } else if (patterns[i].type == "spike") {
                factor += patterns[i].strength * 5; // one-off jump
                patterns[i].remainingTicks = 0;
            } else if (patterns[i].type == "crash") {
                factor -= patterns[i].strength * 5; // one-off drop
                patterns[i].remainingTicks = 0;
            } else if (patterns[i].type == "oscillate") {
                factor += std::sin((tick % 10) / 10.0 * 2 * M_PI) * patterns[i].strength;
            }

            patterns[i].remainingTicks--;
        }

        double newPrice = eq.getPrice() * factor;
        if (newPrice < 0.01) newPrice = 0.01;

        // Correlation rules
        if (eq.getSymbol() == "TSLA" && uni(rng) < 0.7) {
            double aaplChange = equities[0].getPrice() / newPrice - 1.0;
            newPrice *= (1.0 + aaplChange * 0.8);
        }
        if (eq.getSymbol() == "GOOG" && uni(rng) < 0.6) {
            double amznChange = equities[2].getPrice() / newPrice - 1.0;
            newPrice *= (1.0 + amznChange * 0.75);
        }

        // Add random noise
        double volatilityNoise = 1.0 + ((uni(rng) - 0.5) * eq.getVolatility() * 4);
        newPrice *= volatilityNoise;
        if (newPrice < 0.01) newPrice = 0.01;

        // Update equity directly
        eq.setPrice(newPrice);
        int newVolume = eq.getVolume() + static_cast<int>((uni(rng) - 0.5) * 2000);
        if (newVolume < 0) newVolume = 0;
        eq.setVolume(newVolume);
    }

    tick++;
}

std::string Market::getMarketDataJSON() const {
    std::ostringstream oss;
    oss << "{ \"tick\": " << tick << ", \"equities\": [";
    for (size_t i = 0; i < equities.size(); i++) {
        oss << "{ \"symbol\": \"" << equities[i].getSymbol() << "\", "
            << "\"price\": " << equities[i].getPrice() << ", "
            << "\"volume\": " << equities[i].getVolume() << "}";
        if (i != equities.size() - 1) oss << ",";
    }
    oss << "] }";
    return oss.str();
}

bool Market::executeTrade(const std::string &userId, const std::string &symbol, const std::string &side, int quantity) {
    for (auto &eq : equities) {
        if (eq.getSymbol() == symbol) {
            Trade trade;
            trade.userId = userId;
            trade.symbol = symbol;
            trade.side = side;
            trade.quantity = quantity;
            trade.price = eq.getPrice();
            trade.timestamp = std::time(nullptr);
            tradeBook.recordTrade(trade);
            return true;
        }
    }
    return false;
}

const TradeBook& Market::getTradeBook() const {
    return tradeBook;
}