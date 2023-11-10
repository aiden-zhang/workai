






#ifndef DL_MACRO_H
#define DL_MACRO_H

#include <sstream>
#include <iostream>

#include "dl_log.h"
#include "dl_timer.h"

#define call(call_func, dl_timer, warmup_times, excute_times)               \
    do {                                                                    \
        for (int i = 0; i < warmup_times; i++) {                            \
            call_func;                                                      \
        }                                                                   \
                                                                            \
        dl_timer->start();                                                  \
        for(int i = 0; i < excute_times; i++) {                             \
            call_func;                                                      \
        }                                                                   \
        cudaDeviceSynchronize();                                            \
        dl_timer->stop();                                                   \
                                                                            \
        DlLogI << "It takes " << dl_timer->last_elapsed() / excute_times    \
               << " ms average on " <<  excute_times << " times with "      \
               << warmup_times << " warm ups!";                             \
    } while (0);

template <class Type>
Type stringToNumber(const std::string& s)
{
    std::istringstream iss(s);
    Type num;
    iss >> num;
    return num;
}

template <class Type>
std::string numberToString(Type number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}

#endif //DL_MACRO_H
