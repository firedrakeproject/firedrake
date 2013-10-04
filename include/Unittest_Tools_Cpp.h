#ifndef UNITTEST_TOOLS_CPP_H
#define UNITTEST_TOOLS_CPP_H

#include <iostream>
#include <string>
#include <limits>

#include "confdefs.h"

void report_test(const std::string& title, const bool& fail, const bool& warn, const std::string& msg);

extern "C" {
void cget_nan(double* nan);
}

#endif
