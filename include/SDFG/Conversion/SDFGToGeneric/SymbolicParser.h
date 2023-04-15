#ifndef SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
#define SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::sdfg::conversion {

class Node {
public:
  virtual ~Node() = default;
  virtual mlir::Value codegen(mlir::PatternRewriter &rewriter,
                              mlir::Location loc) = 0;
  virtual void collect_variables(SmallVector<std::string> &variables) = 0;
};

class SymbolicParser {
public:
  static std::unique_ptr<Node> parse(StringRef input, unsigned pos = 0);
};

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
