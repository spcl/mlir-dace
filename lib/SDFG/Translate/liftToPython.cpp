#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"

using namespace mlir;
using namespace sdfg;

// TODO(later): Temporary auto-lifting. Will be included into DaCe
Optional<std::string> liftOperationToPython(Operation &op, Operation &source) {
  std::string nameOut = op.getNumResults() == 1
                            ? utils::valueToString(op.getResult(0), op)
                            : "Not Supported";
  //===--------------------------------------------------------------------===//
  // Arith
  //===--------------------------------------------------------------------===//

  if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
    std::string nameArg0 = utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " + " + nameArg1;
  }

  if (isa<arith::SubFOp>(op) || isa<arith::SubIOp>(op)) {
    std::string nameArg0 = utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " - " + nameArg1;
  }

  if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
    std::string nameArg0 = utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " * " + nameArg1;
  }

  if (arith::DivFOp divFOp = dyn_cast<arith::DivFOp>(op)) {
    return nameOut + " = " + utils::valueToString(divFOp.getLhs(), op) + " / " +
           utils::valueToString(divFOp.getRhs(), op);
  }

  if (arith::NegFOp negFOp = dyn_cast<arith::NegFOp>(op)) {
    return nameOut + " = -" + utils::valueToString(negFOp.getOperand(), op);
  }

  if (arith::RemSIOp remSIOp = dyn_cast<arith::RemSIOp>(op)) {
    return nameOut + " = " + utils::valueToString(remSIOp.getLhs(), op) +
           " % " + utils::valueToString(remSIOp.getRhs(), op);
  }

  if (arith::IndexCastOp indexCast = dyn_cast<arith::IndexCastOp>(op)) {
    return nameOut + " = " + utils::valueToString(indexCast.getIn(), op);
  }

  if (arith::SIToFPOp sIToFPOp = dyn_cast<arith::SIToFPOp>(op)) {
    return nameOut + " = float(" + utils::valueToString(sIToFPOp.getIn(), op) +
           ")";
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

    std::string lhs = utils::valueToString(lhsValue, op);
    std::string rhs = utils::valueToString(rhsValue, op);

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
    return nameOut + " = " + utils::valueToString(selectOp.getTrueValue(), op) +
           " if " + utils::valueToString(selectOp.getCondition(), op) +
           " else " + utils::valueToString(selectOp.getFalseValue(), op);
  }

  //===--------------------------------------------------------------------===//
  // Math
  //===--------------------------------------------------------------------===//

  if (math::SqrtOp sqrtOp = dyn_cast<math::SqrtOp>(op)) {
    return nameOut + " = math.sqrt(" +
           utils::valueToString(sqrtOp.getOperand(), op) + ")";
  }

  if (math::ExpOp expOp = dyn_cast<math::ExpOp>(op)) {
    return nameOut + " = math.exp(" +
           utils::valueToString(expOp.getOperand(), op) + ")";
  }

  if (math::PowFOp powFOp = dyn_cast<math::PowFOp>(op)) {
    return nameOut + " = math.pow(" +
           utils::valueToString(powFOp.getLhs(), op) + "," +
           utils::valueToString(powFOp.getRhs(), op) + ")";
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
    return nameOut + " = " + sym.expr().str();
  }

  if (StoreOp store = dyn_cast<StoreOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(utils::valueToString(op.getOperand(i), op));
    }

    std::string nameVal = utils::valueToString(store.arr(), op);
    return nameOut + "[" + indices + "]" + " = " + nameVal;
  }

  if (LoadOp load = dyn_cast<LoadOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(utils::valueToString(op.getOperand(i), op));
    }

    std::string nameArr = utils::valueToString(load.arr(), op);
    return nameOut + " = " + nameArr + "[" + indices + "]";
  }

  if (StreamLengthOp streamLen = dyn_cast<StreamLengthOp>(op)) {
    // NOTE: What's the proper stream name?
    std::string streamName = utils::valueToString(streamLen.str(), op);
    return nameOut + " = len(" + streamName + ")";
  }

  if (isa<sdfg::ReturnOp>(op)) {
    std::string code = "";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (TaskletNode task = dyn_cast<TaskletNode>(source)) {
        if (i > 0)
          code.append("\\n");
        code.append(task.getOutputName(i) + " = " +
                    utils::valueToString(op.getOperand(i), op));
        continue;
      }

      // Tasklets are the only ones using sdfg.return
      return None;
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
      // NOTE: What's the proper return name?
      code.append("_out = " + utils::valueToString(op.getOperand(i), op));
    }
    return code;
  }

  return None;
}

// If successful returns Python code as string
Optional<std::string> translation::liftToPython(Operation &op) {
  std::string code = "";

  for (Operation &oper : op.getRegion(0).getOps()) {
    Optional<std::string> line = liftOperationToPython(oper, op);
    if (line.hasValue()) {
      code.append(line.getValue() + "\\n");
    } else {
      emitRemark(op.getLoc(), "No lifting to python possible");
      emitRemark(oper.getLoc(), "Failed to lift");
      return None;
    }
  }

  return code;
}

std::string translation::getTaskletName(Operation &op) {
  Operation &firstOp = *op.getRegion(0).getOps().begin();
  return utils::operationToString(firstOp);
}
