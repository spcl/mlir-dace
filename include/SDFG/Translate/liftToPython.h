#ifndef SDFG_Translation_LiftToPython_H
#define SDFG_Translation_LiftToPython_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::translation {

Optional<std::string> liftToPython(TaskletNode &op);
std::string getTaskletName(TaskletNode &op);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_LiftToPython_H
