#include "SDFG/Conversion/SDFGToGeneric/SymbolicParser.h"
#include <regex>

using namespace mlir;
using namespace sdfg::conversion;

// Implements a simple LR(1) parser for symbolic expressions

/*
grammar SymExpr;

// Entry point
expr
  : assignment
  | logical_expr
  | arith_expr
  ;

assignment
  : variable ASSIGN arith_expr
  ;

logical_expr
  : logical_term (LOGICAL_OR logical_term)*
  ;

logical_term
  : comparison (LOGICAL_AND comparison)*
  ;

comparison
  : bitwise_or_expression (comparison_operator bitwise_or_expression)?
  ;

bitwise_or_expression
  : bitwise_xor_expression (BITWISE_OR bitwise_xor_expression)*
  ;

bitwise_xor_expression
  : bitwise_and_expression (BITWISE_XOR bitwise_and_expression)*
  ;

bitwise_and_expression
  : shift_expression (BITWISE_AND shift_expression)*
  ;

shift_expression
  : arith_expr (SHIFT_LEFT arith_expr | SHIFT_RIGHT arith_expr)*
  ;

arith_expr
  : term (ADD term | SUB term)*
  ;

term
  : factor (MUL factor | DIV factor | FLOORDIV factor | MOD factor)*
  ;

factor
  : primary (EXP primary)?
  ;

primary
  : integer_constant
  | boolean_constant
  | variable
  | unary_operator primary
  | LPAREN logical_expr RPAREN
  | LPAREN arith_expr RPAREN
  ;

integer_constant
  : DIGIT+
  ;

boolean_constant
  : TRUE
  | FALSE
  ;

variable
  : IDENTIFIER
  ;

comparison_operator
  : EQ | NE | LT | LE | GT | GE
  ;

unary_operator
  : ADD | SUB | NOT
  ;

// Tokens
EQ          : '==';
NE          : '!=';
LT          : '<';
LE          : '<=';
GT          : '>';
GE          : '>=';
ASSIGN      : ':';
LOGICAL_OR  : 'or';
LOGICAL_AND : 'and';
ADD         : '+';
SUB         : '-';
MUL         : '*';
DIV         : '/';
FLOORDIV    : '//';
MOD         : '%';
EXP         : '**';
NOT         : 'not';
TRUE        : 'True';
FALSE       : 'False';
BITWISE_OR  : '|';
BITWISE_XOR : '^';
BITWISE_AND : '&';
SHIFT_LEFT  : '<<';
SHIFT_RIGHT : '>>';
LPAREN      : '(';
RPAREN      : ')';
DIGIT       : [0-9];
IDENTIFIER  : [_a-zA-Z] [_a-zA-Z0-9]*;
WS          : [ \t\r\n]+ -> skip;

*/
namespace mlir::sdfg::conversion {

//===----------------------------------------------------------------------===//
// AST Nodes
//===----------------------------------------------------------------------===//

class IntegerNode : public ASTNode {
public:
  int value;

  IntegerNode(int value) : value(value) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for constants
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {}
};

class BooleanNode : public ASTNode {
public:
  bool value;

  BooleanNode(bool value) : value(value) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for constants
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {}
};

class VariableNode : public ASTNode {
public:
  std::string name;

  VariableNode(const std::string &name) : name(name) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for variables
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    variables.push_back(name);
  }
};

class AssignmentNode : public ASTNode {
public:
  std::unique_ptr<VariableNode> variable;
  std::unique_ptr<ASTNode> expr;

  AssignmentNode(std::unique_ptr<VariableNode> variable,
                 std::unique_ptr<ASTNode> expr)
      : variable(std::move(variable)), expr(std::move(expr)) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for assignments
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    variable->collect_variables(variables);
    expr->collect_variables(variables);
  }
};

class UnOpNode : public ASTNode {
public:
  enum Op { ADD, SUB, NOT };
  UnOpNode::Op op;
  std::unique_ptr<ASTNode> expr;

  UnOpNode(UnOpNode::Op op, std::unique_ptr<ASTNode> expr)
      : op(op), expr(std::move(expr)) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for unary operators
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    expr->collect_variables(variables);
  }
};

class BinOpNode : public ASTNode {
public:
  enum Op {
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
    SHIFT_LEFT,
    SHIFT_RIGHT,
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
  BinOpNode::Op op;
  std::unique_ptr<ASTNode> right;

  BinOpNode(std::unique_ptr<ASTNode> left, BinOpNode::Op op,
            std::unique_ptr<ASTNode> right)
      : left(std::move(left)), op(op), right(std::move(right)) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for binary operators
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    left->collect_variables(variables);
    right->collect_variables(variables);
  }
};

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

enum TokenType {
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE,
  ASSIGN,
  LOGICAL_OR,
  LOGICAL_AND,
  ADD,
  SUB,
  MUL,
  DIV,
  FLOORDIV,
  MOD,
  EXP,
  NOT,
  TRUE,
  FALSE,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_AND,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  LPAREN,
  RPAREN,
  INTEGER_CONSTANT,
  IDENTIFIER,
  WS
};

struct Token {
  TokenType type;
  std::string value;
};

Optional<SmallVector<Token>> tokenize(StringRef input) {
  llvm::StringMap<TokenType> tokenDefinitions = {
      {"==", EQ},
      {"!=", NE},
      {"<", LT},
      {"<=", LE},
      {">", GT},
      {">=", GE},
      {":", ASSIGN},
      {"or", LOGICAL_OR},
      {"and", LOGICAL_AND},
      {"\\+", ADD},
      {"-", SUB},
      {"\\*", MUL},
      {"/", DIV},
      {"//", FLOORDIV},
      {"%", MOD},
      {"\\*\\*", EXP},
      {"not", NOT},
      {"True", TRUE},
      {"False", FALSE},
      {"\\|", BITWISE_OR},
      {"\\^", BITWISE_XOR},
      {"&", BITWISE_AND},
      {"<<", SHIFT_LEFT},
      {">>", SHIFT_RIGHT},
      {"\\(", LPAREN},
      {"\\)", RPAREN},
      {"\\d+", INTEGER_CONSTANT},
      {"[_a-zA-Z][_a-zA-Z0-9]*", IDENTIFIER},
      {"[ \\t\\r\\n]+", WS}};

  SmallVector<Token> tokens;
  std::string remaining = input.str();

  while (!remaining.empty()) {
    bool matched = false;

    for (llvm::StringMapEntry<TokenType> &definition : tokenDefinitions) {
      std::regex pattern("^" + definition.first().str());
      std::smatch match;
      if (std::regex_search(remaining, match, pattern)) {
        if (definition.second != WS) // Skip whitespace tokens
          tokens.push_back(Token{definition.second, match.str()});

        remaining = match.suffix().str();
        matched = true;
        break;
      }
    }

    if (!matched)
      return std::nullopt;
  }

  return tokens;
}

//===----------------------------------------------------------------------===//
// Parsing Table
//===----------------------------------------------------------------------===/

enum Action { SHIFT, REDUCE, GOTO };

struct TableEntry {
  Action action;
  int value;
};

Optional<TableEntry> get_action_entry(int state, Token lookahead) {}

Optional<TableEntry> get_goto_entry(int state) {}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

std::unique_ptr<ASTNode> SymbolicParser::parse(StringRef input) {
  Optional<SmallVector<Token>> tokens = tokenize(input);
  if (!tokens.has_value())
    return nullptr;

  std::unique_ptr<ASTNode> node;

  return nullptr;
}

} // namespace mlir::sdfg::conversion
