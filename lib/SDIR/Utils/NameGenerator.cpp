#include "SDIR/Utils/NameGenerator.h"

namespace mlir::sdir::utils {
namespace {
int nameGeneratorID = 0;
}

std::string generateName(std::string base) {
  return base + "_" + std::to_string(nameGeneratorID++);
}

} // namespace mlir::sdir::utils
