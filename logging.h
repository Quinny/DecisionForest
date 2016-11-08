#ifndef LOGGING_H
#define LOGGING_H

#include <ctime>
#include <iostream>
#include <cstring>

namespace qp {
namespace logging {

bool enabled = false;

struct LogStream {
  template <typename T>
  std::ostream& operator<<(const T& x) {
    const auto tm = std::time(nullptr);
    const auto* curtime = localtime(&tm);
    auto* time_str = asctime(curtime);
    time_str[std::strlen(time_str) - 1] = 0;  // stupid new line.
    std::cout << time_str << " -- " << x;
    return std::cout;
  }
};

}  // namespace logging

logging::LogStream LOG;

}  // namespace qp

#endif /* LOGGING_H */
