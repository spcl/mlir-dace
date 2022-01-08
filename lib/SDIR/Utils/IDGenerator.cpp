#include "SDIR/Utils/IDGenerator.h"

namespace mlir::sdir::utils {
namespace {
int idGeneratorID = 0;
}

int generateID() { return idGeneratorID++; }

void resetIDGenerator() { idGeneratorID = 0; }

} // namespace mlir::sdir::utils
