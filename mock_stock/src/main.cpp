#include "Market.h"
#include "Equity.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <set>

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib") // Link Winsock library

#define PORT 5000

std::set<SOCKET> clients;
std::mutex clientsMutex;

void broadcastToClients(const std::string &message) {
    std::lock_guard<std::mutex> lock(clientsMutex);
    for (auto sock : clients) {
        send(sock, message.c_str(), (int)message.size(), 0);
    }
}

void clientHandler(SOCKET clientSocket, Market &market) {
    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        clients.insert(clientSocket);
        std::cout << "New client connected. Total clients: " << clients.size() << std::endl;
    }

    char buffer[1024];
    while (true) {
        memset(buffer, 0, sizeof(buffer));
        int bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0) break;

        std::string command(buffer);
        if (command.rfind("TRADE", 0) == 0) {
            std::istringstream iss(command);
            std::string cmd, userId, symbol, side;
            int qty;
            iss >> cmd >> userId >> symbol >> side >> qty;
            bool success = market.executeTrade(userId, symbol, side, qty);
            std::string resp = success ? "TRADE_OK\n" : "TRADE_FAIL\n";
            send(clientSocket, resp.c_str(), (int)resp.size(), 0);
        } else if (command.rfind("GET_TRADES", 0) == 0) {
            std::istringstream iss(command);
            std::string cmd, userId;
            iss >> cmd >> userId;
            const auto &trades = market.getTradeBook().getTradesForUser(userId);
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < trades.size(); i++) {
                oss << "{ \"symbol\": \"" << trades[i].symbol << "\", "
                    << "\"side\": \"" << trades[i].side << "\", "
                    << "\"qty\": " << trades[i].quantity << ", "
                    << "\"price\": " << trades[i].price << "}";
                if (i != trades.size() - 1) oss << ",";
            }
            oss << "]\n";
            send(clientSocket, oss.str().c_str(), (int)oss.str().size(), 0);
        }
    }

    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        clients.erase(clientSocket);
        std::cout << "Client disconnected. Total clients: " << clients.size() << std::endl;
    }
    closesocket(clientSocket);
}

int main() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        return 1;
    }

    Market market;
    market.addEquity(Equity("AAPL", 150.0, 0.02, 1000000));
    market.addEquity(Equity("TSLA", 700.0, 0.04, 500000));
    market.addEquity(Equity("AMZN", 3300.0, 0.015, 800000));
    market.addEquity(Equity("GOOG", 2800.0, 0.018, 600000));

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "Socket creation failed." << std::endl;
        WSACleanup();
        return 1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "Bind failed." << std::endl;
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    if (listen(server_fd, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "Listen failed." << std::endl;
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    std::cout << "Server listening on port " << PORT << "..." << std::endl;

    std::thread marketThread([&]() {
        while (true) {
            market.step();
            std::string data = market.getMarketDataJSON() + "\n";
            std::cout << "Market Tick Sent: " << data << std::endl; // ✅ now always visible
            broadcastToClients(data);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    while (true) {
        SOCKET clientSocket = accept(server_fd, nullptr, nullptr);
        if (clientSocket != INVALID_SOCKET) {
            std::thread(clientHandler, clientSocket, std::ref(market)).detach();
        }
    }

    marketThread.join();
    closesocket(server_fd);
    WSACleanup();
    return 0;
}