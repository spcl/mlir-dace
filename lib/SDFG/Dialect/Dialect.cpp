#include "SDFG/Dialect/Dialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sdfg;

#include "SDFG/Dialect/OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SDFG Dialect
//===----------------------------------------------------------------------===//

void SDFGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SDFG/Dialect/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "SDFG/Dialect/OpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SDFG Types
//===----------------------------------------------------------------------===//

// TODO: Rewrite to only use an ArrayAttr containing strings & ints
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

::mlir::Type ArrayType::parse(::mlir::AsmParser &odsParser) {
  Type elementType;
  SmallVector<StringAttr> symbols;
  SmallVector<int64_t> integers;
  SmallVector<bool> shape;
  if (parseDimensionList(odsParser, elementType, symbols, integers, shape))
    return Type();

  SizedType sized = SizedType::get(odsParser.getContext(), elementType, symbols,
                                   integers, shape);
  return get(odsParser.getContext(), sized);
}

void ArrayType::print(::mlir::AsmPrinter &odsPrinter) const {
  Type elemType = getDimensions().getElementType();
  ArrayRef<StringAttr> symbols = getDimensions().getSymbols();
  ArrayRef<int64_t> integers = getDimensions().getIntegers();
  ArrayRef<bool> shape = getDimensions().getShape();

  printDimensionList(odsPrinter, elemType, symbols, integers, shape);
}

::mlir::Type StreamType::parse(::mlir::AsmParser &odsParser) {
  Type elementType;
  SmallVector<StringAttr> symbols;
  SmallVector<int64_t> integers;
  SmallVector<bool> shape;
  if (parseDimensionList(odsParser, elementType, symbols, integers, shape))
    return Type();

  SizedType sized = SizedType::get(odsParser.getContext(), elementType, symbols,
                                   integers, shape);
  return get(odsParser.getContext(), sized);
}

void StreamType::print(::mlir::AsmPrinter &odsPrinter) const {
  Type elemType = getDimensions().getElementType();
  ArrayRef<StringAttr> symbols = getDimensions().getSymbols();
  ArrayRef<int64_t> integers = getDimensions().getIntegers();
  ArrayRef<bool> shape = getDimensions().getShape();

  printDimensionList(odsPrinter, elemType, symbols, integers, shape);
}

#define GET_TYPEDEF_CLASSES
#include "SDFG/Dialect/OpsTypes.cpp.inc"