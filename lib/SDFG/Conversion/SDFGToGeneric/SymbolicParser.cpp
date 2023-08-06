// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Conversion/SDFGToGeneric/SymbolicParser.h"
#include "SDFG/Conversion/SDFGToGeneric/OpCreators.h"
#include <regex>

using namespace mlir;
using namespace sdfg::conversion;

// Implements a simple LL(1) parser for symbolic expressions

/*
stmt         ::= assignment | log_or_expr
assignment   ::= IDENT ASSIGN log_or_expr

log_or_expr  ::= log_and_expr ( LOG_OR log_and_expr )*
log_and_expr ::= eq_expr ( LOG_AND eq_expr )*

eq_expr      ::= rel_expr ( ( EQ | NE ) rel_expr )*
rel_expr     ::= shift_expr ( ( LT | LE | GT | GE ) shift_expr )*
shift_expr   ::= bit_or_expr ( ( LSHIFT | RSHIFT ) bit_or_expr )*

bit_or_expr  ::= bit_xor_expr ( BIT_OR bit_xor_expr )*
bit_xor_expr ::= bit_and_expr ( BIT_XOR bit_and_expr )*
bit_and_expr ::= add_expr ( BIT_AND add_expr )*

add_expr     ::= mul_expr ( ( ADD | SUB ) mul_expr )*
mul_expr     ::= exp_expr ( ( MUL | DIV | FLOORDIV | MOD ) exp_expr )*
exp_expr     ::= unary_expr ( EXP unary_expr )*
unary_expr   ::= ( ADD | SUB | LOG_NOT | BIT_NOT )? factor
factor       ::= LPAREN log_or_expr RPAREN | const_expr | IDENT

const_expr   ::= bool_const | INT_CONST
bool_const   ::= TRUE | FALSE

// Tokens
EQ          ::= '==';
NE          ::= '!=';
LT          ::= '<';
LE          ::= '<=';
GT          ::= '>';
GE          ::= '>=';
ASSIGN      ::= ':';
LOG_OR      ::= 'or';
LOG_AND     ::= 'and';
LOG_NOT     ::= 'not';
ADD         ::= '+';
SUB         ::= '-';
MUL         ::= '*';
DIV         ::= '/';
FLOORDIV    ::= '//';
MOD         ::= '%';
EXP         ::= '**';
TRUE        ::= 'True';
FALSE       ::= 'False';
BIT_OR      ::= '|';
BIT_XOR     ::= '^';
BIT_AND     ::= '&';
BIT_NOT     ::= '~';
LSHIFT      ::= '<<';
RSHIFT      ::= '>>';
LPAREN      ::= '(';
RPAREN      ::= ')';
INT_CONST   ::= DIGIT+
IDENT       ::= LETTER ( LETTER | DIGIT )*
WS          ::= [ \t\r\n]+ -> skip;

// Helpers
DIGIT       ::= [0-9];
LETTER      ::= [_a-zA-Z];

*/

namespace mlir::sdfg::conversion {

//===----------------------------------------------------------------------===//
// AST Nodes
//===----------------------------------------------------------------------===//

/// Converts the integer node into MLIR code. SymbolMap is used for permanent
/// mapping of symbols to values. RefMap is a temporary mapping overriding
/// SymbolMap.
Value IntNode::codegen(PatternRewriter &rewriter, Location loc,
                       llvm::StringMap<Value> &symbolMap,
                       llvm::StringMap<Value> &refMap) {
  return createConstantInt(rewriter, loc, value, 64);
}

/// Converts the boolean node into MLIR code. SymbolMap is used for permanent
/// mapping of symbols to values. RefMap is a temporary mapping overriding
/// SymbolMap.
Value BoolNode::codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<Value> &symbolMap,
                        llvm::StringMap<Value> &refMap) {
  return createConstantInt(rewriter, loc, value, 1);
}

/// Converts the variable node into MLIR code. SymbolMap is used for permanent
/// mapping of symbols to values. RefMap is a temporary mapping overriding
/// SymbolMap.
Value VarNode::codegen(PatternRewriter &rewriter, Location loc,
                       llvm::StringMap<Value> &symbolMap,
                       llvm::StringMap<Value> &refMap) {
  if (refMap.find(name) != refMap.end()) {
    Value val = refMap[name];

    if (val.getType().isIndex())
      return createIndexCast(rewriter, loc, rewriter.getI64Type(), val);

    if (val.getType().isIntOrIndex() &&
        val.getType().getIntOrFloatBitWidth() != 64)
      return createExtSI(rewriter, loc, rewriter.getI64Type(), val);

    return val;
  }

  allocSymbol(rewriter, loc, name, symbolMap);
  return createLoad(rewriter, loc, symbolMap[name], {});
}

/// Converts the assignment node into MLIR code. SymbolMap is used for
/// permanent mapping of symbols to values. RefMap is a temporary mapping
/// overriding SymbolMap.
Value AssignNode::codegen(PatternRewriter &rewriter, Location loc,
                          llvm::StringMap<Value> &symbolMap,
                          llvm::StringMap<Value> &refMap) {
  allocSymbol(rewriter, loc, variable->name, symbolMap);
  Value eVal = expr->codegen(rewriter, loc, symbolMap, refMap);
  createStore(rewriter, loc, eVal, symbolMap[variable->name], {});
  return nullptr;
}

/// Converts the unary operation node into MLIR code. SymbolMap is used for
/// permanent mapping of symbols to values. RefMap is a temporary mapping
/// overriding SymbolMap.
Value UnOpNode::codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<Value> &symbolMap,
                        llvm::StringMap<Value> &refMap) {
  Value eVal = expr->codegen(rewriter, loc, symbolMap, refMap);

  switch (op) {
  case ADD:
    return eVal;
  case SUB: {
    Value zero = createConstantInt(rewriter, loc, 0, 64);
    return createSubI(rewriter, loc, zero, eVal);
  }
  case LOG_NOT: {
    Value zero = createConstantInt(rewriter, loc, 0, 1);
    return createCmpI(rewriter, loc, arith::CmpIPredicate::eq, zero, eVal);
  }
  case BIT_NOT: {
    Value negOne = createConstantInt(rewriter, loc, -1, 64);
    return createXOrI(rewriter, loc, negOne, eVal);
  }
  }

  return eVal;
}

/// Converts the binary operation node into MLIR code. SymbolMap is used for
/// permanent mapping of symbols to values. RefMap is a temporary mapping
/// overriding SymbolMap.
Value BinOpNode::codegen(PatternRewriter &rewriter, Location loc,
                         llvm::StringMap<Value> &symbolMap,
                         llvm::StringMap<Value> &refMap) {
  Value lVal = left->codegen(rewriter, loc, symbolMap, refMap);
  Value rVal = right->codegen(rewriter, loc, symbolMap, refMap);

  switch (op) {
  case ADD:
    return createAddI(rewriter, loc, lVal, rVal);
  case SUB:
    return createSubI(rewriter, loc, lVal, rVal);
  case MUL:
    return createMulI(rewriter, loc, lVal, rVal);
  case DIV:
    return createDivSI(rewriter, loc, lVal, rVal);
  case FLOORDIV:
    return createFloorDivSI(rewriter, loc, lVal, rVal);
  case MOD:
    return createRemSI(rewriter, loc, lVal, rVal);
  case EXP:
    break;
  // TODO: Implement EXP case
  case BIT_OR:
    return createOrI(rewriter, loc, lVal, rVal);
  case BIT_XOR:
    return createXOrI(rewriter, loc, lVal, rVal);
  case BIT_AND:
    return createAndI(rewriter, loc, lVal, rVal);
  case LSHIFT:
    return createShLI(rewriter, loc, lVal, rVal);
  case RSHIFT:
    return createShRSI(rewriter, loc, lVal, rVal);
  case LOG_OR:
    break;
  // TODO: Implement LOG_OR case
  case LOG_AND:
    break;
  // TODO: Implement LOG_AND case
  case EQ:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::eq, lVal, rVal);
  case NE:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::ne, lVal, rVal);
  case LT:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::slt, lVal, rVal);
  case LE:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::sle, lVal, rVal);
  case GT:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::sgt, lVal, rVal);
  case GE:
    return createCmpI(rewriter, loc, arith::CmpIPredicate::sge, lVal, rVal);
  }

  return lVal;
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

/// Converts the symbolic expression to individual tokens.
Optional<SmallVector<Token>> SymbolicParser::tokenize(StringRef input) {
  std::vector<std::pair<std::string, TokenType>> tokenDefinitions = {
      {"==", EQ},
      {"!=", NE},
      {"<<", LSHIFT},
      {">>", RSHIFT},
      {"<=", LE},
      {">=", GE},
      {"<", LT},
      {">", GT},
      {":", ASSIGN},
      {"or", LOG_OR},
      {"and", LOG_AND},
      {"not", LOG_NOT},
      {"\\+", ADD},
      {"-", SUB},
      {"\\*\\*", EXP},
      {"\\*", MUL},
      {"//", FLOORDIV},
      {"/", DIV},
      {"%", MOD},
      {"True", TRUE},
      {"False", FALSE},
      {"\\|", BIT_OR},
      {"\\^", BIT_XOR},
      {"&", BIT_AND},
      {"~", BIT_NOT},
      {"\\(", LPAREN},
      {"\\)", RPAREN},
      {"\\d+", INT_CONST},
      {"[_a-zA-Z][_a-zA-Z0-9]*", IDENT},
      {"[ \\t\\r\\n]+", WS}};

  SmallVector<Token> tokens;
  std::string remaining = input.str();

  while (!remaining.empty()) {
    bool matched = false;

    for (std::pair<std::string, TokenType> &definition : tokenDefinitions) {
      std::regex pattern("^" + definition.first);
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
// Parser
//===----------------------------------------------------------------------===//

/// Attempts to parse a statement:
/// stmt ::= assignment | log_or_expr
std::unique_ptr<ASTNode> SymbolicParser::stmt() {
  std::unique_ptr<ASTNode> assignNode = assignment();

  if (assignNode != nullptr)
    return assignNode;

  return log_or_expr();
}

/// Attempts to parse an assignment:
/// assignment ::= IDENT ASSIGN log_or_expr
std::unique_ptr<ASTNode> SymbolicParser::assignment() {
  if (pos + 2 >= tokens.size() || tokens[pos].type != TokenType::IDENT ||
      tokens[pos + 1].type != TokenType::ASSIGN)
    return nullptr;

  std::string varName = tokens[pos].value;
  pos += 2;
  std::unique_ptr<ASTNode> orNode = log_or_expr();

  if (orNode == nullptr)
    return nullptr;

  std::unique_ptr<VarNode> varNode = std::make_unique<VarNode>(varName);
  return std::make_unique<AssignNode>(std::move(varNode), std::move(orNode));
}

/// Attempts to parse a logical OR expression:
/// log_or_expr ::= log_and_expr ( LOG_OR log_and_expr )*
std::unique_ptr<ASTNode> SymbolicParser::log_or_expr() {
  std::unique_ptr<ASTNode> leftNode = log_and_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::LOG_OR) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = log_and_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::LOG_OR, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse a logical AND expression:
/// log_and_expr ::= eq_expr ( LOG_AND eq_expr )*
std::unique_ptr<ASTNode> SymbolicParser::log_and_expr() {
  std::unique_ptr<ASTNode> leftNode = eq_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::LOG_AND) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = eq_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::LOG_AND, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an equality expression:
/// eq_expr ::= rel_expr ( ( EQ | NE ) rel_expr )*
std::unique_ptr<ASTNode> SymbolicParser::eq_expr() {
  std::unique_ptr<ASTNode> leftNode = rel_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && (tokens[pos].type == TokenType::EQ ||
                                     tokens[pos].type == TokenType::NE)) {
    BinOpNode::BinOp binOp;

    switch (tokens[pos].type) {
    case TokenType::EQ:
      binOp = BinOpNode::BinOp::EQ;
      break;
    case TokenType::NE:
      binOp = BinOpNode::BinOp::NE;
      break;
    default:
      break;
    }

    pos++;
    std::unique_ptr<ASTNode> rightNode = rel_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(std::move(leftNode), binOp,
                                           std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an inequality expression:
/// rel_expr ::= shift_expr ( ( LT | LE | GT | GE ) shift_expr )*
std::unique_ptr<ASTNode> SymbolicParser::rel_expr() {
  std::unique_ptr<ASTNode> leftNode = shift_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && (tokens[pos].type == TokenType::LT ||
                                     tokens[pos].type == TokenType::LE ||
                                     tokens[pos].type == TokenType::GT ||
                                     tokens[pos].type == TokenType::GE)) {
    BinOpNode::BinOp binOp;

    switch (tokens[pos].type) {
    case TokenType::LT:
      binOp = BinOpNode::BinOp::LT;
      break;
    case TokenType::LE:
      binOp = BinOpNode::BinOp::LE;
      break;
    case TokenType::GT:
      binOp = BinOpNode::BinOp::GT;
      break;
    case TokenType::GE:
      binOp = BinOpNode::BinOp::GE;
      break;
    default:
      break;
    }

    pos++;
    std::unique_ptr<ASTNode> rightNode = shift_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(std::move(leftNode), binOp,
                                           std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse a shift expression:
/// shift_expr ::= bit_or_expr ( (LSHIFT | RSHIFT ) bit_or_expr )*
std::unique_ptr<ASTNode> SymbolicParser::shift_expr() {
  std::unique_ptr<ASTNode> leftNode = bit_or_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && (tokens[pos].type == TokenType::LSHIFT ||
                                     tokens[pos].type == TokenType::RSHIFT)) {
    BinOpNode::BinOp binOp;

    switch (tokens[pos].type) {
    case TokenType::LSHIFT:
      binOp = BinOpNode::BinOp::LSHIFT;
      break;
    case TokenType::RSHIFT:
      binOp = BinOpNode::BinOp::RSHIFT;
      break;
    default:
      break;
    }

    pos++;
    std::unique_ptr<ASTNode> rightNode = bit_or_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(std::move(leftNode), binOp,
                                           std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse a bitwise OR expression:
/// bit_or_expr ::= bit_xor_expr ( BIT_OR bit_xor_expr )*
std::unique_ptr<ASTNode> SymbolicParser::bit_or_expr() {
  std::unique_ptr<ASTNode> leftNode = bit_xor_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_OR) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = bit_xor_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::BIT_OR, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse a bitwise XOR expression:
/// bit_xor_expr ::= bit_and_expr ( BIT_XOR bit_and_expr )*
std::unique_ptr<ASTNode> SymbolicParser::bit_xor_expr() {
  std::unique_ptr<ASTNode> leftNode = bit_and_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_XOR) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = bit_and_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::BIT_XOR, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse a bitwise AND expression:
/// bit_and_expr ::= add_expr ( BIT_AND add_expr )*
std::unique_ptr<ASTNode> SymbolicParser::bit_and_expr() {
  std::unique_ptr<ASTNode> leftNode = add_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_AND) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = add_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::BIT_AND, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an arithmetic addition / subtraction expression:
/// add_expr ::= mul_expr ( ( ADD | SUB ) mul_expr )*
std::unique_ptr<ASTNode> SymbolicParser::add_expr() {
  std::unique_ptr<ASTNode> leftNode = mul_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && (tokens[pos].type == TokenType::ADD ||
                                     tokens[pos].type == TokenType::SUB)) {
    BinOpNode::BinOp binOp;

    switch (tokens[pos].type) {
    case TokenType::ADD:
      binOp = BinOpNode::BinOp::ADD;
      break;
    case TokenType::SUB:
      binOp = BinOpNode::BinOp::SUB;
      break;
    default:
      break;
    }

    pos++;
    std::unique_ptr<ASTNode> rightNode = mul_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(std::move(leftNode), binOp,
                                           std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an arithmetic multiplication / division / floor / modulo
/// expression:
/// mul_expr ::= exp_expr ( ( MUL | DIV | FLOORDIV | MOD ) exp_expr )*
std::unique_ptr<ASTNode> SymbolicParser::mul_expr() {
  std::unique_ptr<ASTNode> leftNode = exp_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && (tokens[pos].type == TokenType::MUL ||
                                     tokens[pos].type == TokenType::DIV ||
                                     tokens[pos].type == TokenType::FLOORDIV ||
                                     tokens[pos].type == TokenType::MOD)) {
    BinOpNode::BinOp binOp;

    switch (tokens[pos].type) {
    case TokenType::MUL:
      binOp = BinOpNode::BinOp::MUL;
      break;
    case TokenType::DIV:
      binOp = BinOpNode::BinOp::DIV;
      break;
    case TokenType::FLOORDIV:
      binOp = BinOpNode::BinOp::FLOORDIV;
      break;
    case TokenType::MOD:
      binOp = BinOpNode::BinOp::MOD;
      break;
    default:
      break;
    }

    pos++;
    std::unique_ptr<ASTNode> rightNode = exp_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(std::move(leftNode), binOp,
                                           std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an arithmetic exponential expression:
/// exp_expr ::= unary_expr ( EXP unary_expr )*
std::unique_ptr<ASTNode> SymbolicParser::exp_expr() {
  std::unique_ptr<ASTNode> leftNode = unary_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::EXP) {
    pos++;
    std::unique_ptr<ASTNode> rightNode = unary_expr();

    if (rightNode == nullptr)
      return nullptr;

    leftNode = std::make_unique<BinOpNode>(
        std::move(leftNode), BinOpNode::BinOp::EXP, std::move(rightNode));
  }

  return leftNode;
}

/// Attempts to parse an unary positive / negative / logical and bitwise NOT
/// expression:
/// unary_expr ::= ( ADD | SUB | LOG_NOT | BIT_NOT )? factor
std::unique_ptr<ASTNode> SymbolicParser::unary_expr() {
  if (pos >= tokens.size())
    return nullptr;

  UnOpNode::UnOp unop;
  switch (tokens[pos].type) {
  case TokenType::ADD:
    unop = UnOpNode::UnOp::ADD;
    break;
  case TokenType::SUB:
    unop = UnOpNode::UnOp::SUB;
    break;
  case TokenType::LOG_NOT:
    unop = UnOpNode::UnOp::LOG_NOT;
    break;
  case TokenType::BIT_NOT:
    unop = UnOpNode::UnOp::BIT_NOT;
    break;
  default:
    return factor();
  }

  pos++;
  std::unique_ptr<ASTNode> node = factor();
  if (node == nullptr)
    return nullptr;

  return std::make_unique<UnOpNode>(unop, std::move(node));
}

/// Attempts to parse a single factor:
/// factor ::= LPAREN log_or_expr RPAREN | const_expr | IDENT
std::unique_ptr<ASTNode> SymbolicParser::factor() {
  if (pos >= tokens.size())
    return nullptr;

  if (tokens[pos].type == TokenType::LPAREN) {
    pos++;
    std::unique_ptr<ASTNode> expr = log_or_expr();
    if (expr == nullptr)
      return nullptr;

    if (pos >= tokens.size() || tokens[pos].type != TokenType::RPAREN)
      return nullptr;

    pos++;
    return expr;
  }

  std::unique_ptr<ASTNode> constExpr = const_expr();
  if (constExpr != nullptr)
    return constExpr;

  if (tokens[pos].type == TokenType::IDENT) {
    std::unique_ptr<VarNode> varNode =
        std::make_unique<VarNode>(tokens[pos].value);
    pos++;
    return varNode;
  }

  return nullptr;
}

/// Attempts to parse a constant expression:
/// const_expr ::= bool_const | INT_CONST
std::unique_ptr<ASTNode> SymbolicParser::const_expr() {
  if (pos >= tokens.size())
    return nullptr;

  std::unique_ptr<ASTNode> boolExpr = bool_const();
  if (boolExpr != nullptr)
    return boolExpr;

  if (tokens[pos].type == TokenType::INT_CONST) {
    std::unique_ptr<IntNode> intNode =
        std::make_unique<IntNode>(std::stoi(tokens[pos].value));
    pos++;
    return intNode;
  }

  return nullptr;
}

/// Attempts to parse a constant boolean expression:
/// bool_const ::= TRUE | FALSE
std::unique_ptr<ASTNode> SymbolicParser::bool_const() {
  if (pos >= tokens.size())
    return nullptr;

  if (tokens[pos].type == TokenType::TRUE) {
    pos++;
    return std::make_unique<BoolNode>(true);
  }

  if (tokens[pos].type == TokenType::FALSE) {
    pos++;
    return std::make_unique<BoolNode>(false);
  }

  return nullptr;
}

/// Parses a symbolic expression provided as a string to an AST.
std::unique_ptr<ASTNode> SymbolicParser::parse(StringRef input) {
  Optional<SmallVector<Token>> tokens = tokenize(input);
  if (!tokens.has_value())
    return nullptr;

  this->tokens = tokens.value();
  this->pos = 0;
  return stmt();
}

} // namespace mlir::sdfg::conversion
