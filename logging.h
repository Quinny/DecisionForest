#ifndef LOGGING_H
#define LOGGING_H

#include <cstring>
#include <ctime>
#include <iostream>

/*
 * Logging helpers.
 */

namespace qp {
namespace logging {

bool enabled = true;

// Write the current datetime followed by the user input to stdout.
struct LogStream {
  template <typename T>
  std::ostream& operator<<(const T& x) {
    if (!enabled) return std::cout;

    const auto tm = std::time(nullptr);
    const auto* curtime = localtime(&tm);
    auto* time_str = asctime(curtime);
    time_str[std::strlen(time_str) - 1] = 0;  // stupid new line.
    std::cout << time_str << " -- " << x;
    return std::cout;
  }
};

}  // namespace logging

// Default logging stream to be used.
logging::LogStream LOG;

}  // namespace qp

#endif /* LOGGING_H */
