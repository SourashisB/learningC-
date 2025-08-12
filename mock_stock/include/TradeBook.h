#ifndef TRADEBOOK_H
#define TRADEBOOK_H

#include "Trade.h"
#include <map>
#include <vector>
#include <string>

class TradeBook {
private:
    std::map<std::string, std::vector<Trade>> tradesByUser;

public:
    void recordTrade(const Trade &trade);
    const std::vector<Trade>& getTradesForUser(const std::string &userId) const;
};

#endif