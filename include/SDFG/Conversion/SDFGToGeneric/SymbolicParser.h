// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
#define SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::sdfg::conversion {

/// Parent class representing any AST node.
class ASTNode {
public:
  virtual ~ASTNode() = default;

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  virtual Value codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<Value> &symbolMap,
                        llvm::StringMap<Value> &refMap) = 0;
};

/// Integer AST node representing an integer constant.
class IntNode : public ASTNode {
public:
  int value;

  IntNode(int value) : value(value){};

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Boolean AST node representing a boolean constant.
class BoolNode : public ASTNode {
public:
  bool value;

  BoolNode(bool value) : value(value) {}

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Variable AST node representing a symbol.
class VarNode : public ASTNode {
public:
  std::string name;

  VarNode(std::string name) : name(name) {}

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Assignment AST node representing the assignment of an expression to a
/// variable.
class AssignNode : public ASTNode {
public:
  std::unique_ptr<VarNode> variable;
  std::unique_ptr<ASTNode> expr;

  AssignNode(std::unique_ptr<VarNode> variable, std::unique_ptr<ASTNode> expr)
      : variable(std::move(variable)), expr(std::move(expr)) {}

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Unary Operation AST node representing an unary operation performed on an
/// expression.
class UnOpNode : public ASTNode {
public:
  /// Enum representing all possible unary operations.
  enum UnOp { ADD, SUB, LOG_NOT, BIT_NOT };

  UnOp op;
  std::unique_ptr<ASTNode> expr;

  UnOpNode(UnOp op, std::unique_ptr<ASTNode> expr)
      : op(op), expr(std::move(expr)) {}

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Binary Operation AST node representing a binary operation performed on an
/// expression.
class BinOpNode : public ASTNode {
public:
  /// Enum representing all possible binary operations.
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

  /// Converts the node into MLIR code. SymbolMap is used for permanent mapping
  /// of symbols to values. RefMap is a temporary mapping overriding SymbolMap.
  Value codegen(PatternRewriter &rewriter, Location loc,
                llvm::StringMap<Value> &symbolMap,
                llvm::StringMap<Value> &refMap) override;
};

/// Enum representing all accepted token types.
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

/// Struct to assign a token type to each parsed token.
struct Token {
  TokenType type;
  std::string value;
};

/// This parser parses symbolic expressions. The private functions attempt to
/// parse one specific grammar rule in descending precedence order.
class SymbolicParser {
public:
  /// Parses a symbolic expression provided as a string to an AST.
  std::unique_ptr<ASTNode> parse(StringRef input);

private:
  unsigned pos;
  SmallVector<Token> tokens;

  /// Converts the symbolic expression to individual tokens.
  Optional<SmallVector<Token>> tokenize(StringRef input);

  /// Attempts to parse a statement:
  /// stmt ::= assignment | log_or_expr.
  std::unique_ptr<ASTNode> stmt();
  /// Attempts to parse an assignment:
  /// assignment ::= IDENT ASSIGN log_or_expr.
  std::unique_ptr<ASTNode> assignment();

  /// Attempts to parse a logical OR expression:
  /// log_or_expr ::= log_and_expr ( LOG_OR log_and_expr )*
  std::unique_ptr<ASTNode> log_or_expr();
  /// Attempts to parse a logical AND expression:
  /// log_and_expr ::= eq_expr ( LOG_AND eq_expr )*
  std::unique_ptr<ASTNode> log_and_expr();

  /// Attempts to parse an equality expression:
  /// eq_expr ::= rel_expr ( ( EQ | NE ) rel_expr )*
  std::unique_ptr<ASTNode> eq_expr();
  /// Attempts to parse an inequality expression:
  /// rel_expr ::= shift_expr ( ( LT | LE | GT | GE ) shift_expr )*
  std::unique_ptr<ASTNode> rel_expr();
  /// Attempts to parse a shift expression:
  /// shift_expr ::= bit_or_expr ( (LSHIFT | RSHIFT ) bit_or_expr )*
  std::unique_ptr<ASTNode> shift_expr();

  /// Attempts to parse a bitwise OR expression:
  /// bit_or_expr ::= bit_xor_expr ( BIT_OR bit_xor_expr )*
  std::unique_ptr<ASTNode> bit_or_expr();
  /// Attempts to parse a bitwise XOR expression:
  /// bit_xor_expr ::= bit_and_expr ( BIT_XOR bit_and_expr )*
  std::unique_ptr<ASTNode> bit_xor_expr();
  /// Attempts to parse a bitwise AND expression:
  /// bit_and_expr ::= add_expr ( BIT_AND add_expr )*
  std::unique_ptr<ASTNode> bit_and_expr();

  /// Attempts to parse an arithmetic addition / subtraction expression:
  /// add_expr ::= mul_expr ( ( ADD | SUB ) mul_expr )*
  std::unique_ptr<ASTNode> add_expr();
  /// Attempts to parse an arithmetic multiplication / division / floor / modulo
  /// expression:
  /// mul_expr ::= exp_expr ( ( MUL | DIV | FLOORDIV | MOD ) exp_expr )*
  std::unique_ptr<ASTNode> mul_expr();
  /// Attempts to parse an arithmetic exponential expression:
  /// exp_expr ::= unary_expr ( EXP unary_expr )*
  std::unique_ptr<ASTNode> exp_expr();
  /// Attempts to parse an unary positive / negative / logical and bitwise NOT
  /// expression:
  /// unary_expr ::= ( ADD | SUB | LOG_NOT | BIT_NOT )? factor
  std::unique_ptr<ASTNode> unary_expr();
  /// Attempts to parse a single factor:
  /// factor ::= LPAREN log_or_expr RPAREN | const_expr | IDENT
  std::unique_ptr<ASTNode> factor();

  /// Attempts to parse a constant expression:
  /// const_expr ::= bool_const | INT_CONST
  std::unique_ptr<ASTNode> const_expr();
  /// Attempts to parse a constant boolean expression:
  /// bool_const ::= TRUE | FALSE
  std::unique_ptr<ASTNode> bool_const();
};

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Symbolic_Parser_H
