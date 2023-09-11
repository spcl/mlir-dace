// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains a Python lifter, which lifts MLIR operations to Python
/// code.

#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"

using namespace mlir;
using namespace sdfg;

// TODO(later): Temporary auto-lifting. Will be included into DaCe
/// Converts a single operation to a single line of Python code. If successful,
/// returns Python code as s string.
Optional<std::string> liftOperationToPython(Operation &op, Operation &source) {
  // FIXME: Support multiple return values
  if (op.getNumResults() > 1)
    return std::string("Not Supported");

  std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);

  bool taskletToSingle = true;
  if (TaskletNode task = dyn_cast<TaskletNode>(source)) {
    // Figure out if we can transform the tasklet into a one-liner, i.e. no
    // interdependent operations (one use = return op)
    for (Operation &operation : task.getOps()) {
      if (!operation.hasOneUse() && operation.getNumResults() > 0) {
        taskletToSingle = false;
        break;
      }
    }

    // If we can, change the output name to the tasklet connector
    if (taskletToSingle) {
      unsigned i = 0;
      for (Operation &operation : task.getOps()) {
        if (operation.getLoc() == op.getLoc())
          break;
        ++i;
      }

      nameOut = task.getOutputName(i);
    }
  }

  //===--------------------------------------------------------------------===//
  // Arith
  //===--------------------------------------------------------------------===//

  if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " + " + nameArg1;
  }

  if (isa<arith::SubFOp>(op) || isa<arith::SubIOp>(op)) {
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " - " + nameArg1;
  }

  if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " * " + nameArg1;
  }

  if (isa<arith::DivFOp>(op) || isa<arith::DivSIOp>(op) ||
      isa<arith::DivUIOp>(op)) {
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " / " + nameArg1;
  }

  if (arith::NegFOp negFOp = dyn_cast<arith::NegFOp>(op)) {
    return nameOut + " = -" +
           sdfg::utils::valueToString(negFOp.getOperand(), op);
  }

  if (isa<arith::RemSIOp>(op) || isa<arith::RemUIOp>(op) ||
      isa<arith::RemFOp>(op)) {
    return nameOut + " = " + sdfg::utils::valueToString(op.getOperand(0), op) +
           " % " + sdfg::utils::valueToString(op.getOperand(1), op);
  }

  if (arith::IndexCastOp indexCast = dyn_cast<arith::IndexCastOp>(op)) {
    return nameOut + " = " + sdfg::utils::valueToString(indexCast.getIn(), op);
  }

  if (isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) {
    return nameOut + " = float(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ")";
  }

  if (isa<arith::FPToSIOp>(op) || isa<arith::FPToUIOp>(op)) {
    return nameOut + " = int(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ")";
  }

  if (isa<arith::MaxFOp>(op) || isa<arith::MaxSIOp>(op) ||
      isa<arith::MaxUIOp>(op)) {
    return nameOut + " = max(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ", " +
           sdfg::utils::valueToString(op.getOperand(1), op) + ")";
  }

  if (isa<arith::CmpIOp>(op) || isa<arith::CmpFOp>(op)) {
    Value lhsValue;
    Value rhsValue;
    std::string predicate = "";

    if (isa<arith::CmpIOp>(op)) {
      arith::CmpIOp cmp = dyn_cast<arith::CmpIOp>(op);
      lhsValue = cmp.getLhs();
      rhsValue = cmp.getRhs();

      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        predicate = "==";
        break;

      case arith::CmpIPredicate::ne:
        predicate = "!=";
        break;

      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        predicate = ">=";
        break;

      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        predicate = ">";
        break;

      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        predicate = "<=";
        break;

      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        predicate = "<";
        break;

      default:
        break;
      }
    }

    else {
      arith::CmpFOp cmp = dyn_cast<arith::CmpFOp>(op);
      lhsValue = cmp.getLhs();
      rhsValue = cmp.getRhs();

      switch (cmp.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        predicate = "==";
        break;

      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        predicate = "!=";
        break;

      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        predicate = ">=";
        break;

      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        predicate = ">";
        break;

      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        predicate = "<=";
        break;

      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        predicate = "<";
        break;

      default:
        break;
      }
    }

    std::string lhs = sdfg::utils::valueToString(lhsValue, op);
    std::string rhs = sdfg::utils::valueToString(rhsValue, op);

    return nameOut + " = " + lhs + " " + predicate + " " + rhs;
  }

  if (isa<arith::ConstantOp>(op)) {
    std::string val;

    if (arith::ConstantFloatOp flop = dyn_cast<arith::ConstantFloatOp>(op)) {
      SmallVector<char> flopVec;
      flop.value().toString(flopVec);
      for (char c : flopVec)
        val += c;
    } else if (arith::ConstantIntOp iop = dyn_cast<arith::ConstantIntOp>(op)) {
      val = std::to_string(iop.value());
    } else if (arith::ConstantIndexOp iop =
                   dyn_cast<arith::ConstantIndexOp>(op)) {
      val = std::to_string(iop.value());
    }

    return nameOut + " = " + val;
  }

  if (arith::SelectOp selectOp = dyn_cast<arith::SelectOp>(op)) {
    return nameOut + " = " +
           sdfg::utils::valueToString(selectOp.getTrueValue(), op) + " if " +
           sdfg::utils::valueToString(selectOp.getCondition(), op) + " else " +
           sdfg::utils::valueToString(selectOp.getFalseValue(), op);
  }

  if (isa<arith::ExtSIOp>(op) || isa<arith::ExtUIOp>(op) ||
      isa<arith::ExtFOp>(op)) {
    return nameOut + " = " + sdfg::utils::valueToString(op.getOperand(0), op);
  }

  //===--------------------------------------------------------------------===//
  // Math
  //===--------------------------------------------------------------------===//

  if (math::SqrtOp sqrtOp = dyn_cast<math::SqrtOp>(op)) {
    return nameOut + " = math.sqrt(" +
           sdfg::utils::valueToString(sqrtOp.getOperand(), op) + ")";
  }

  if (math::ExpOp expOp = dyn_cast<math::ExpOp>(op)) {
    return nameOut + " = math.exp(" +
           sdfg::utils::valueToString(expOp.getOperand(), op) + ")";
  }

  if (math::PowFOp powFOp = dyn_cast<math::PowFOp>(op)) {
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);

    return nameOut + " = math.pow(" + nameArg0 + "," + nameArg1 + ")";
  }

  if (math::CosOp cosOp = dyn_cast<math::CosOp>(op)) {
    return nameOut + " = math.cos(" +
           sdfg::utils::valueToString(cosOp.getOperand(), op) + ")";
  }

  if (math::SinOp sinOp = dyn_cast<math::SinOp>(op)) {
    return nameOut + " = math.sin(" +
           sdfg::utils::valueToString(sinOp.getOperand(), op) + ")";
  }

  if (math::LogOp logOp = dyn_cast<math::LogOp>(op)) {
    return nameOut + " = math.log(" +
           sdfg::utils::valueToString(logOp.getOperand(), op) + ")";
  }

  //===--------------------------------------------------------------------===//
  // LLVM
  //===--------------------------------------------------------------------===//

  if (isa<mlir::LLVM::UndefOp>(op)) {
    return nameOut + " = -1";
  }

  //===--------------------------------------------------------------------===//
  // SDFG
  //===--------------------------------------------------------------------===//

  if (SymOp sym = dyn_cast<SymOp>(op)) {
    return nameOut + " = " + sym.getExpr().str();
  }

  if (StoreOp store = dyn_cast<StoreOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(sdfg::utils::valueToString(op.getOperand(i), op));
    }

    std::string nameVal = sdfg::utils::valueToString(store.getArr(), op);
    return nameOut + "[" + indices + "]" + " = " + nameVal;
  }

  if (LoadOp load = dyn_cast<LoadOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(sdfg::utils::valueToString(op.getOperand(i), op));
    }

    std::string nameArr = sdfg::utils::valueToString(load.getArr(), op);
    return nameOut + " = " + nameArr + "[" + indices + "]";
  }

  if (StreamLengthOp streamLen = dyn_cast<StreamLengthOp>(op)) {
    // FIXME: What's the proper stream name?
    std::string streamName = sdfg::utils::valueToString(streamLen.getStr(), op);
    return nameOut + " = len(" + streamName + ")";
  }

  if (isa<sdfg::ReturnOp>(op)) {
    std::string code = "";

    // Only add return code if we're not transforming tasklets to one-liners
    if (!taskletToSingle) {
      if (!isa<TaskletNode>(source)) {
        // Tasklets are the only ones using sdfg.return
        return std::nullopt;
      }

      TaskletNode task = cast<TaskletNode>(source);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        if (i > 0)
          code.append("\\n");
        code.append(task.getOutputName(i) + " = " +
                    sdfg::utils::valueToString(op.getOperand(i), op));
      }
    }

    return code;
  }

  //===--------------------------------------------------------------------===//
  // Func
  //===--------------------------------------------------------------------===//

  if (isa<func::ReturnOp>(op)) {
    std::string code = "";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0)
        code.append("\\n");
      // FIXME: What's the proper return name?
      code.append("_out = " + sdfg::utils::valueToString(op.getOperand(i), op));
    }
    return code;
  }

  return std::nullopt;
}

/// Converts the operations in the first region of op to Python code. If
/// successful, returns Python code as a string.
Optional<std::string> translation::liftToPython(Operation &op) {
  std::string code = "";

  for (Operation &oper : op.getRegion(0).getOps()) {
    Optional<std::string> line = liftOperationToPython(oper, op);
    if (line.has_value()) {
      code.append(line.value() + "\\n");
    } else {
      emitRemark(op.getLoc(), "No lifting to python possible");
      emitRemark(oper.getLoc(), "Failed to lift");
      return std::nullopt;
    }
  }

  return code;
}

/// Provides a name for the tasklet.
std::string translation::getTaskletName(Operation &op) {
  Operation &firstOp = *op.getRegion(0).getOps().begin();
  return sdfg::utils::operationToString(firstOp);
}
