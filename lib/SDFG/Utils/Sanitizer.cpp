#include "SDFG/Utils/Sanitizer.h"

using namespace mlir;
using namespace sdfg;

void utils::sanitizeName(std::string &name) {
  for (unsigned i = 0; i < name.size(); ++i) {
    if (!(name[i] >= 'a' && name[i] <= 'z') &&
        !(name[i] >= 'A' && name[i] <= 'Z') &&
        !(name[i] >= '0' && name[i] <= '9')) {
      name[i] = '_';
    }
  }
}
