#include "SDFG/Translate/liftToPython.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace sdfg;

// TODO(later): Temporary auto-lifting. Will be included into DaCe
// If successful returns Python code as string
Optional<std::string> translation::liftToPython(TaskletNode &op) {
  int numOps = 0;
  Operation *firstOp = nullptr;

  for (Operation &oper : op.body().getOps()) {
    if (numOps == 0)
      firstOp = &oper;
    ++numOps;
  }

  if (numOps > 2) {
    emitRemark(op.getLoc(), "No lifting to python possible");
    return None;
  }

  if (isa<arith::AddFOp>(firstOp) || isa<arith::AddIOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    std::string nameArg1 = op.getInputName(op.body().getNumArguments() - 1);
    return "__out = " + nameArg0 + " + " + nameArg1;
  }

  if (isa<arith::MulFOp>(firstOp) || isa<arith::MulIOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    std::string nameArg1 = op.getInputName(op.body().getNumArguments() - 1);
    return "__out = " + nameArg0 + " * " + nameArg1;
  }

  // TODO: Add arith ops

  if (arith::ConstantOp oper = dyn_cast<arith::ConstantOp>(firstOp)) {
    std::string val;

    if (arith::ConstantFloatOp flop =
            dyn_cast<arith::ConstantFloatOp>(firstOp)) {
      SmallVector<char> flopVec;
      flop.value().toString(flopVec);
      for (char c : flopVec)
        val += c;
    } else if (arith::ConstantIntOp iop =
                   dyn_cast<arith::ConstantIntOp>(firstOp)) {
      val = std::to_string(iop.value());
    } else if (arith::ConstantIndexOp iop =
                   dyn_cast<arith::ConstantIndexOp>(firstOp)) {
      val = std::to_string(iop.value());
    }

    return "__out = " + val;
  }

  if (arith::IndexCastOp ico = dyn_cast<arith::IndexCastOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    return "__out = " + nameArg0;
  }

  if (StoreOp store = dyn_cast<StoreOp>(firstOp)) {
    std::string indices;

    for (unsigned i = 0; i < op.body().getNumArguments() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(op.getInputName(i));
    }

    std::string valName = op.getInputName(op.body().getNumArguments() - 1);
    return "__out[" + indices + "]" + " = " + valName;
  }

  if (isa<LoadOp>(firstOp)) {
    std::string indices;

    for (unsigned i = 0; i < op.body().getNumArguments() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(op.getInputName(i));
    }

    std::string arrName = op.getInputName(op.body().getNumArguments() - 1);
    return "__out = " + arrName + "[" + indices + "]";
  }

  if (SymOp sym = dyn_cast<SymOp>(firstOp)) {
    return "__out = " + sym.expr().str();
  }

  if (isa<sdfg::ReturnOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    return "__out = " + nameArg0;
  }

  emitRemark(op.getLoc(), "No lifting to python possible");
  return None;
}

std::string translation::getTaskletName(TaskletNode &op) {
  Operation &firstOp = *op.body().getOps().begin();

  if (isa<arith::AddFOp>(firstOp) || isa<arith::AddIOp>(firstOp))
    return "add";
  else if (isa<arith::MulFOp>(firstOp) || isa<arith::MulIOp>(firstOp))
    return "mult";
  else if (isa<arith::ConstantOp>(firstOp))
    return "constant";
  else if (isa<arith::IndexCastOp>(firstOp))
    return "cast";
  else if (isa<StoreOp>(firstOp))
    return "store";
  else if (isa<LoadOp>(firstOp))
    return "load";
  else if (isa<SymOp>(firstOp))
    return "sym ";
  else if (isa<sdfg::ReturnOp>(firstOp))
    return "return";

  return "task";
}
