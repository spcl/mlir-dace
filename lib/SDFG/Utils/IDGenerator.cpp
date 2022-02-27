#include "SDFG/Utils/IDGenerator.h"

namespace mlir::sdfg::utils {
namespace {
unsigned idGeneratorID = 0;
}

unsigned generateID() { return idGeneratorID++; }

void resetIDGenerator() { idGeneratorID = 0; }

} // namespace mlir::sdfg::utils
