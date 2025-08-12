#include "Market.h"
#include <sstream>
#include <ctime>

Market::Market() : tick(0) {}

void Market::addEquity(const Equity &eq) {
    equities.push_back(eq);
}

void Market::step() {
    for (auto &eq : equities) {
        eq.updatePrice();
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