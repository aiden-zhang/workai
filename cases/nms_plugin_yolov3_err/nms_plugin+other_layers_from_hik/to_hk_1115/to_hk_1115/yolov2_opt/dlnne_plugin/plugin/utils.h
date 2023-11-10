
#pragma once
#include <assert.h>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "cuda.h"
#include "dlnne.h"

using namespace std;
#define SHOW_DEBUG_INFO 0

template <typename T>
const void DebugInfo(T info) {
#if SHOW_DEBUG_INFO
  std::cout << info << std::endl;
#endif
}

template <typename T, typename... Ts>
const void DebugInfo(T info, Ts... infos) {
#if SHOW_DEBUG_INFO
  std::cout << info << ",";

  DebugInfo<Ts...>(infos...);

#endif
}

template <typename T>
struct remove_const_impl {
  typedef T type;
};
template <typename T>
struct remove_const_impl<T*> {
  typedef T type;
};
template <typename T>
struct remove_const_impl<const T*> {
  typedef T type;
};

template <typename T>
void PrintfCudaMemory(T data, int size, int cols = 32) {
#if SHOW_DEBUG_INFO
  typedef typename remove_const_impl<T>::type DataType;

  DataType* host_data = (DataType*)malloc(sizeof(DataType) * size);

  cudaMemcpy(host_data, data, sizeof(DataType) * size, cudaMemcpyDeviceToHost);

  std::cout << "printf_cudaMemory:[" << std::endl;
  for (int i = 0; i < size; i++) {
    if (host_data[i] != 0) {
      std::cout << host_data[i] << ",";
    } else {
      continue;
    }
    if (i % cols == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << "]" << std::endl;

  free(host_data);
#endif
}

// file operations
bool DLFileExists(const char* path);

bool DLMkdir(const char* path);

void DLFileRemove(const char* path);

bool DLDirectoryRemove(const char* path, bool delete_self = false);

class CacheDirMgr {
 public:
  CacheDirMgr() {}

  ~CacheDirMgr() { RemoveTmpFile(); }

  bool Init() {
    if (cache_dir_inited_) return true;

    cache_dir_inited_ = true;
    char* tmp_dir = getenv("DLCI_CACHE_DIR");

    char username[256];
    getlogin_r(username, sizeof(username));

    if (tmp_dir) {
      snprintf(cache_topdir_, kMaxFilenameLength_ - 1, "%s", tmp_dir);
    } else {
      snprintf(cache_topdir_, kMaxFilenameLength_ - 1, "/tmp/.%s/", username);
    }

    if (!DLFileExists(cache_topdir_)) {
      // always make dir
      if (!DLMkdir(cache_topdir_)) {
        cache_dir_inited_ = false;
        assert(false);
      }
    }

    return cache_dir_inited_;
  }

  bool GenTmpFileName(std::string& filename, const char* postfix) {
    if (!cache_dir_inited_) return false;

    char file_name[kMaxFilenameLength_] = {0};
    int bytes_written = snprintf(file_name, kMaxFilenameLength_ - 1,
                                 "%s/temp_XXXXXX.%s", cache_topdir_, postfix);

    assert(bytes_written > 0 && bytes_written < kMaxFilenameLength_ - 1);

    int fd = mkstemps(file_name, strlen(postfix) + 1);
    if (fd <= 0) {
      assert(false);
    }
    close(fd);

    // save the created file name
    tmp_files_.push_back(file_name);
    filename = file_name;

    return true;
  }

  bool RemoveTmpFile() {
    // remove temp files
    for (auto iter : tmp_files_) {
      if (DLFileExists(iter.c_str())) {
        DLFileRemove(iter.c_str());
      }
    }

    return true;
  }

  const char* GetCacheTopDir() { return cache_topdir_; }

 private:
  static const int kMaxFilenameLength_{1024};
  char cache_topdir_[kMaxFilenameLength_];
  bool cache_dir_inited_{false};

  std::vector<std::string> tmp_files_;
};
