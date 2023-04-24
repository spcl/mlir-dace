#ifndef SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
#define SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::sdfg::conversion {

class ASTNode {
public:
  virtual ~ASTNode() = default;
  virtual Value codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<memref::AllocOp> &symbolMap) = 0;
};

class IntNode : public ASTNode {
public:
  int value;

  IntNode(int value) : value(value){};

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
};

class BoolNode : public ASTNode {
public:
  bool value;

  BoolNode(bool value) : value(value) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
};

class VarNode : public ASTNode {
public:
  std::string name;

  VarNode(std::string name) : name(name) {}

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
};

class AssignNode : public ASTNode {
public:
  VarNode *variable;
  ASTNode *expr;

  AssignNode(VarNode *variable, ASTNode *expr)
      : variable(variable), expr(expr) {}

  ~AssignNode() {
    delete variable;
    delete expr;
  }

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
};

class UnOpNode : public ASTNode {
public:
  enum UnOp { ADD, SUB, LOG_NOT, BIT_NOT };
  UnOp op;
  ASTNode *expr;

  UnOpNode(UnOp op, ASTNode *expr) : op(op), expr(expr) {}
  ~UnOpNode() { delete expr; }

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
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

  ASTNode *left;
  BinOp op;
  ASTNode *right;

  BinOpNode(ASTNode *left, BinOp op, ASTNode *right)
      : left(left), op(op), right(right) {}

  ~BinOpNode() {
    delete left;
    delete right;
  }

  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<memref::AllocOp> &symbolMap) override;
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
  // FIXME: Replace bare pointers with smart pointers
  ASTNode *parse(StringRef input);

private:
  unsigned pos;
  SmallVector<Token> tokens;

  Optional<SmallVector<Token>> tokenize(StringRef input);

  ASTNode *stmt();
  ASTNode *assignment();

  ASTNode *log_or_expr();
  ASTNode *log_and_expr();

  ASTNode *eq_expr();
  ASTNode *rel_expr();
  ASTNode *shift_expr();

  ASTNode *bit_or_expr();
  ASTNode *bit_xor_expr();
  ASTNode *bit_and_expr();

  ASTNode *add_expr();
  ASTNode *mul_expr();
  ASTNode *exp_expr();
  ASTNode *unary_expr();
  ASTNode *factor();

  ASTNode *const_expr();
  ASTNode *bool_const();
};

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
