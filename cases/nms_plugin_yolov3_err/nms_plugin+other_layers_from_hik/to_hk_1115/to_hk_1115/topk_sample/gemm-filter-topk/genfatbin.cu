#include "./topk/topk_interface.h"
#include "./topk/generation.h"

#ifdef __DLGPUC64__
#include <cu_ext.h>
#endif

#include <stdlib.h>
using namespace std;

void gen_all_case(){

	const std::vector<int> K_vec={16, 32, 64, 128, 256, 512, 1024};
	std::vector<int> datasize_vec;
    for(int i=0;i<22;i++){

		datasize_vec.push_back(1024*pow(2, i));
	}

	for(int K: K_vec){
		for(auto datasize: datasize_vec){
			if(datasize>=16*K)
				GenerateCase(K, datasize);
		}
	}
}

bool cat_kernels() {
    std::string cmd = "cat /tmp/topk/*.cu 2>&1 | tee /tmp/kernels.cu > /dev/null 2>&1";
    std::string cmd2 = "chmod 777 -R /tmp/topk && chmod 777 /tmp/kernels.cu";

    if (system(cmd.c_str()) || system(cmd2.c_str())) {
      	return false;
    }else{
        return true;
    }

}

bool compile_cases_to_fatbin(){

	string source_file_name_ = "/tmp/kernels.cu";

	std::string target_file_name = "kernels.fatbin";
	std::string bc_file_name = "kernels.bc";
	bool success = true;

	#ifndef SRC_DIR
	#define SRC_DIR "-I./"
	#endif
        std::string inc_dir(SRC_DIR);
	#ifndef __DLGPUC64__
	    std::string cmd = "nvcc -arch=sm_61 --fatbin -O3  -std=c++14 " + inc_dir +" " + source_file_name_ + " -o " + target_file_name;

	    std::cout<<"starting online compile:"<<cmd<<std::endl;
	    success = system(cmd.c_str());

	#else
	    std::stringstream cmd;
	    cmd<<"dlcc --cuda-gpu-arch=dlgpuc64 --cuda-device-only -S -std=c++14 " + inc_dir + " " + source_file_name_ + " -o " + bc_file_name+";";
	    cmd<<"dlvm-link -march=dlgpuc64 " + bc_file_name << " -o " + target_file_name;
	    std::cout<<"starting online compile:"<<cmd.str()<<std::endl;
	    success = system(cmd.str().c_str());

	#endif

    if (success) {
    	std::cout<<"online compile failed"<<std::endl;
      	return false;
    }else{
    	std::cout<<"online compile success"<<std::endl;    	
        return true;
    }
}



int main(int argc, char const *argv[])
{

	ofile.open("/tmp/kernels.cu",ios::out);
	gen_all_case();	
	ofile.close();	
	// if(cat_kernels()){
	bool compile_success = compile_cases_to_fatbin();
	// }

    return 0;
}

