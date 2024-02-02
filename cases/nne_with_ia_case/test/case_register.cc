#include "case_register.h"


using namespace std;

void* test_case[10];
int case_num = 0;
case_register::case_register(void *func) {
    test_case[case_num] = func;
    case_num++;
}

