cmake_minimum_required(VERSION 3.15)
project(MockStockMarket)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(MockStockMarket
    src/main.cpp
    src/Equity.cpp
    src/Market.cpp
    src/TradeBook.cpp
)