#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>


void showUsage()
{
    std::cout << "Usage: ./testnetworkperf [m] [b] [e] [t] [s] [d] [o] [h]\n";
    std::cout << "    m: model\n";
    std::cout << "    b: batchsize\n";
    std::cout << "    e: execution times\n";
    std::cout << "    t: thread nums\n";
    std::cout << "    s: do serialized or not\n";
    std::cout << "    d: do deserialized or not\n";
    std::cout << "    o: output nodes\n";
    std::cout << "    h:  help\n";
    std::cout << "    eg: ./a.out -m ./resnet18.onnx -b 16 -e 100 -s 1 -d 0\n";
    // exit(0);
}

#define TEST_RETURN_FALSE_EXPR(expr)                                         \
  do {                                                                       \
    const bool ret_value = (expr);                                           \
    if (!ret_value) {                                                        \
      printf("Call: %s failed in %s at %d line\n", #expr, __FILE__, __LINE__); \
      return ret_value;                                                      \
    }                                                                        \
  } while (0)

void getCustomOpt(int argc, char *argv[], std::string &model_path, int &max_batch,
                  int &execute_time, int &threadNum,int &is_serialize, int &is_deserialize,
                  std::vector<std::string>& output_nodes)
{
    if(argc < 2) {
        showUsage();
    }
    int opt = 0;
    const char* opt_string = "m:b:e:t:s:d:o:h";
    while(-1 != (opt = getopt(argc, argv, opt_string))) {
        switch (opt) {
        case 'm':
            model_path = optarg;
            break;
        case 'b':
            max_batch = atoi(optarg);
            break;
        case 'e':
            execute_time = atoi(optarg);
            break;
        case 't':
            threadNum = atoi(optarg);
            break;    
        case 's':
            is_serialize = atoi(optarg);
            break;   
        case 'd':
            is_deserialize = atoi(optarg);
            break;                                   
        case 'o':
            output_nodes.push_back(optarg);
            break;
        default:
            showUsage();
            std::cout<<"params err! you can assign -m -b -c"<<std::endl;
            break;
        }
    }
}
