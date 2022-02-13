#include "SDFG/Utils/IDGenerator.h"

namespace mlir::sdfg::utils {
namespace {
int idGeneratorID = 0;
}

int generateID() { return idGeneratorID++; }

void resetIDGenerator() { idGeneratorID = 0; }

} // namespace mlir::sdfg::utils
