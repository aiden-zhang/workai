#ifndef CASE_REGITER_H
#define CASE_REGITER_H
#include <iostream>
#include <map>
#include <vector>

#define NEAREST 0
#define BILINEAR 1
#define FORMAT_YU12    1
#define FORMAT_NV12    2
#define FORMAT_NV21    3

using namespace std;

extern void* test_case[];
extern int case_num;
class case_register {
public:
    case_register(void *func);
};

#endif
