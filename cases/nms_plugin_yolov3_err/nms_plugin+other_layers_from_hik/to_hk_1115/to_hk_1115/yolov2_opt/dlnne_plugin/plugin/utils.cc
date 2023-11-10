#include "utils.h"

// ------------------ file operation ----------------

#define R_OK 4 /* Test for read permission.  */
#define W_OK 2 /* Test for write permission.  */
#define F_OK 0 /* Test for existence.  */

bool DLFileExists(const char* path) { return !access(path, R_OK); }

bool DLMkdir(const char* path) {
  int error = mkdir(path, S_IRWXU);

  if (error && errno == ENOENT) {
    // create directories recursively
    char* prev_path;
    ulong prev_path_length = static_cast<ulong>(strrchr(path, '/') - path);

    prev_path = new char[prev_path_length + 1];
    strncpy(prev_path, path, prev_path_length);
    prev_path[prev_path_length] = '\0';

    DLMkdir(reinterpret_cast<const char*>(prev_path));

    delete[] prev_path;
    error = mkdir(path, S_IRWXU);
  } else if (error && errno == EEXIST) {
    return true;
  } else if (error) {
    return false;
  }

  return true;
}

void DLFileRemove(const char* path) { remove(path); }

bool DLDirectoryRemove(const char* path, bool delete_self) {
  struct stat st_info;
  if (lstat(path, &st_info) == -1) {
    return false;
  }
  if (S_ISREG(st_info.st_mode)) {
    DLFileRemove(path);
  } else if (S_ISDIR(st_info.st_mode)) {
    DIR* dir = opendir(path);
    if (dir == nullptr) {
      return false;
    }
    struct dirent* under_file;
    char file_path[256];
    while (under_file = readdir(dir)) {
      if (strcmp(under_file->d_name, ".") == 0 ||
          strcmp(under_file->d_name, "..") == 0) {
        continue;
      }
      snprintf(file_path, sizeof(file_path), "%s/%s", path, under_file->d_name);
      if (!DLDirectoryRemove(file_path, true)) {
        closedir(dir);
        return false;
      }
    }
    if (delete_self && rmdir(path) == -1) {
      closedir(dir);
      return false;
    }
    closedir(dir);
  }

  return true;
}
