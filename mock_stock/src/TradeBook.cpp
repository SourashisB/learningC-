#include "TradeBook.h"

void TradeBook::recordTrade(const Trade &trade) {
    tradesByUser[trade.userId].push_back(trade);
}

const std::vector<Trade>& TradeBook::getTradesForUser(const std::string &userId) const {
    static std::vector<Trade> empty;
    auto it = tradesByUser.find(userId);
    if (it != tradesByUser.end()) {
        return it->second;
    }
    return empty;
}