#include<stdio.h>
#include<cuda.h>
#include "bin_count.h"
#include <chrono>
#include <set>
#include<iostream>
#include "topk_select.h"
#include <cu_ext.h>

using namespace std;

set<int> fake_data(unsigned char* input_data, int size, int K = 10000) {
	srand(1);
	int pre_bin_num=32;

	for (int i = 0; i < size; i++) {
		unsigned int value = pre_bin_num + (rand() % (256 - pre_bin_num));
		input_data[i] = (unsigned char) value; //put small value on prefix 31 bin,only for test

	}
	set<int> address;

	//write conner case
	for(int i=0;i<5;i++){
		input_data[size-1-i]=(unsigned int) (rand() % pre_bin_num);
		address.insert(size-1-i);
	}

	int k = 5;
	do {
		int ad = rand() % (size-5);
		if (address.find(ad) == address.end()) {
			address.insert(ad);
			input_data[ad] = (unsigned int) (rand() % pre_bin_num);
			k++;
		}
	} while (k < K);

	return address;
}

void print_cuda(int* buffer, int size, int limit_n = 5) {
	int* host_buffer = (int*) malloc(sizeof(int) * size);
	cudaMemcpy(host_buffer, buffer, sizeof(int) * size, cudaMemcpyDeviceToHost);
	std::cout << "printf_cuda_buffer:[";

	for (int i = 0; i < size; i++) {
		if (i % 64 == 0) {
			std::cout << std::endl;
		}
		std::cout << host_buffer[i] << "\t";

		if (i > 64 * limit_n) {
			break;
		}
	}
	std::cout << "]" << std::endl;
}

bool print_check_cuda(int* buffer, int size, set<int> golden) {
	int* host_buffer = (int*) malloc(sizeof(int) * size);
	cudaMemcpy(host_buffer, buffer, sizeof(int) * size, cudaMemcpyDeviceToHost);

	//start check result
	bool success=true;
	for (int i = 0; i < size; i++) {
		int address=host_buffer[i];
		if(golden.find(address)==golden.end()){
			std::cout <<"fail:"<<i<<"th\t" << address<<" not found in golden ......"<<std::endl;
			success=false;
			break;
		}
	}
	if(success){
		std::cout <<"check:Pass..."<<std::endl;
	}else{
		assert(false && "fail.....");
	}
	return success;

}


void test_case(int total_size,int K){
	std::cout<<"total_size:"<<total_size<<" K:"<<K<<std::endl;

	TopkBinSelect<unsigned char> topk(total_size);

	unsigned char* input_data = (unsigned char*) malloc(
			sizeof(char) * total_size);
	auto topk_address_golden = fake_data(input_data, total_size, K);

	unsigned char* input_data_device = nullptr;
	cudaMalloc(&input_data_device, sizeof(char) * total_size);

	int* workspace = nullptr;
	int* topkIdx=nullptr;

	cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);


	cudaMalloc(&workspace,topk.QueryWorkSapce());

	cudaMalloc(&topkIdx, sizeof(int) * (K)); //for finnal result

	cudaMemcpyAsync(input_data_device, input_data, sizeof(char) * total_size,
			cudaMemcpyHostToDevice,stream);

	cudaStreamSynchronize(stream);

	int loop_count=100;
	auto start=chrono::system_clock::now();
	for(int i=0;i<loop_count;i++){
		topk.Execute(input_data_device,K,topkIdx,workspace,stream);
	}

	cudaStreamSynchronize(stream);
	cudaDeviceSynchronize();

	auto end=chrono::system_clock::now();
    auto duration=chrono::duration_cast<chrono::microseconds>(end-start);

	//print_check_cuda(topkIdx, K,topk_address_golden);

	std::cout<<"execte avg time:"<<(duration.count()*1.0f/1000/loop_count)<<"(ms)"<<std::endl;

}
void test_all_case(){

	for(int data_size_log2=16;data_size_log2<28;data_size_log2++){
		int data_size=(1<<data_size_log2);
		data_size=16*(rand()%(data_size/16))+data_size;
		test_case(data_size,10000);
	}

}


int main(int c, char** args) {

    cudaSetClusterMask(1);
	test_case(1024*1024*128/4,10000);
	//test_all_case();
	return 0;
}
