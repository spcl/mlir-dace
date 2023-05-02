#ifndef SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
#define SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::sdfg::conversion {

class ASTNode {
public:
  virtual ~ASTNode() = default;
  virtual Value codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<memref::AllocOp> &symbolMap,
                        llvm::StringMap<Value> &refMap) = 0;
};

class IntNode : public ASTNode {
public:
  int value;

  IntNode(int value) : value(value){};

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

class BoolNode : public ASTNode {
public:
  bool value;

  BoolNode(bool value) : value(value) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

class VarNode : public ASTNode {
public:
  std::string name;

  VarNode(std::string name) : name(name) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

class AssignNode : public ASTNode {
public:
  std::unique_ptr<VarNode> variable;
  std::unique_ptr<ASTNode> expr;

  AssignNode(std::unique_ptr<VarNode> variable, std::unique_ptr<ASTNode> expr)
      : variable(std::move(variable)), expr(std::move(expr)) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

class UnOpNode : public ASTNode {
public:
  enum UnOp { ADD, SUB, LOG_NOT, BIT_NOT };
  UnOp op;
  std::unique_ptr<ASTNode> expr;

  UnOpNode(UnOp op, std::unique_ptr<ASTNode> expr)
      : op(op), expr(std::move(expr)) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

class BinOpNode : public ASTNode {
public:
  enum BinOp {
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    EXP,
    BIT_OR,
    BIT_XOR,
    BIT_AND,
    LSHIFT,
    RSHIFT,
    LOG_OR,
    LOG_AND,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
  };

  std::unique_ptr<ASTNode> left;
  BinOp op;
  std::unique_ptr<ASTNode> right;

  BinOpNode(std::unique_ptr<ASTNode> left, BinOp op,
            std::unique_ptr<ASTNode> right)
      : left(std::move(left)), op(op), right(std::move(right)) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

enum TokenType {
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE,
  ASSIGN,
  LOG_OR,
  LOG_AND,
  LOG_NOT,
  ADD,
  SUB,
  MUL,
  DIV,
  FLOORDIV,
  MOD,
  EXP,
  TRUE,
  FALSE,
  BIT_OR,
  BIT_XOR,
  BIT_AND,
  BIT_NOT,
  LSHIFT,
  RSHIFT,
  LPAREN,
  RPAREN,
  INT_CONST,
  IDENT,
  WS
};

struct Token {
  TokenType type;
  std::string value;
};

class SymbolicParser {
public:
  std::unique_ptr<ASTNode> parse(StringRef input);

private:
  unsigned pos;
  SmallVector<Token> tokens;

  Optional<SmallVector<Token>> tokenize(StringRef input);

  std::unique_ptr<ASTNode> stmt();
  std::unique_ptr<ASTNode> assignment();

  std::unique_ptr<ASTNode> log_or_expr();
  std::unique_ptr<ASTNode> log_and_expr();

  std::unique_ptr<ASTNode> eq_expr();
  std::unique_ptr<ASTNode> rel_expr();
  std::unique_ptr<ASTNode> shift_expr();

  std::unique_ptr<ASTNode> bit_or_expr();
  std::unique_ptr<ASTNode> bit_xor_expr();
  std::unique_ptr<ASTNode> bit_and_expr();

  std::unique_ptr<ASTNode> add_expr();
  std::unique_ptr<ASTNode> mul_expr();
  std::unique_ptr<ASTNode> exp_expr();
  std::unique_ptr<ASTNode> unary_expr();
  std::unique_ptr<ASTNode> factor();

  std::unique_ptr<ASTNode> const_expr();
  std::unique_ptr<ASTNode> bool_const();
};

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
