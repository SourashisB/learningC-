#ifndef MARKET_H
#define MARKET_H

#include "Equity.h"
#include "TradeBook.h"
#include <vector>
#include <string>

class Market {
private:
    std::vector<Equity> equities;
    TradeBook tradeBook;
    int tick;

public:
    Market();

    void addEquity(const Equity &eq);
    void step();
    std::string getMarketDataJSON() const;
    bool executeTrade(const std::string &userId, const std::string &symbol, const std::string &side, int quantity);

    const TradeBook& getTradeBook() const;
};

#endif