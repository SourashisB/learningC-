#ifndef TRADE_H
#define TRADE_H

#include <string>
#include <ctime>

struct Trade {
    std::string userId;
    std::string symbol;
    std::string side; // "BUY" or "SELL"
    int quantity;
    double price;
    std::time_t timestamp;
};

#endif