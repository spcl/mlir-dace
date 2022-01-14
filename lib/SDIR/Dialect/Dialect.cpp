#include "SDIR/Dialect/Dialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sdir;

#include "SDIR/Dialect/OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

void SDIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SDIR/Dialect/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "SDIR/Dialect/OpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

static ParseResult parseDimensionList(AsmParser &parser, Type &elemType,
                                      SmallVector<StringAttr> &symbols,
                                      SmallVector<int64_t> &integers,
                                      SmallVector<bool> &shape) {
  if (parser.parseLess())
    return failure();

  do {
    OptionalParseResult typeOPR = parser.parseOptionalType(elemType);
    if (typeOPR.hasValue()) {
      if (typeOPR.getValue().succeeded()) {
        if (parser.parseGreater())
          return failure();
        return success();
      } else
        return failure();
    }

    if (parser.parseOptionalKeyword("sym").succeeded()) {
      std::string symEx;

      if (parser.parseLParen() || parser.parseString(&symEx) ||
          parser.parseRParen())
        return failure();

      symbols.push_back(parser.getBuilder().getStringAttr(symEx));
      shape.push_back(false);
      continue;
    }

    int64_t num;
    OptionalParseResult intOPR = parser.parseOptionalInteger(num);
    if (intOPR.hasValue()) {
      if (intOPR.getValue().succeeded()) {
        integers.push_back(num);
        shape.push_back(true);
        continue;
      } else
        return failure();
    }

    if (parser.parseOptionalQuestion().succeeded()) {
      integers.push_back(-1);
      shape.push_back(true);
      continue;
    }

    return failure();
  } while (parser.parseXInDimensionList().succeeded());

  return failure();
}

static void printDimensionList(AsmPrinter &printer, Type &elemType,
                               ArrayRef<StringAttr> &symbols,
                               ArrayRef<int64_t> &integers,
                               ArrayRef<bool> &shape) {
  unsigned symIdx = 0;
  unsigned intIdx = 0;

  printer << "<";

  for (unsigned i = 0; i < shape.size(); ++i)
    if (shape[i])
      if (integers[intIdx++] == -1)
        printer << "?x";
      else
        printer << integers[intIdx - 1] << "x";
    else
      printer << "sym(" << symbols[symIdx++] << ")x";

  printer << elemType << ">";
}

#define GET_TYPEDEF_CLASSES
#include "SDIR/Dialect/OpsTypes.cpp.inc"

Type SDIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();

  Type genType;
  OptionalParseResult parseResult =
      generatedTypeParser(parser, mnemonic, genType);
  if (parseResult.hasValue())
    return genType;

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  parser.emitError(typeLoc, "unknown type in SDIR dialect");

  return Type();
}

void SDIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  LogicalResult logRes = generatedTypePrinter(type, os);
  if (logRes.failed())
    emitError(nullptr, "Failed to print dialect type");
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

MemletType ArrayType::toMemlet() {
  return MemletType::get(getContext(), getElementType(), getSymbols(),
                         getIntegers(), getShape());
}

//===----------------------------------------------------------------------===//
// MemletType
//===----------------------------------------------------------------------===//

ArrayType MemletType::toArray() {
  return ArrayType::get(getContext(), getElementType(), getSymbols(),
                        getIntegers(), getShape());
}
