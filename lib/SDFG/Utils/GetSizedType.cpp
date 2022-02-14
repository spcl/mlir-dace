#include "SDFG/Utils/GetSizedType.h"

namespace mlir::sdfg::utils {

SizedType getSizedType(Type t) {
  if (ArrayType arr = t.dyn_cast<ArrayType>())
    return arr.getDimensions();

  return t.cast<StreamType>().getDimensions();
}

} // namespace mlir::sdfg::utils
