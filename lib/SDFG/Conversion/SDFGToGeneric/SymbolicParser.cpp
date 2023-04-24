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

// IntNode
Value IntNode::codegen(PatternRewriter &rewriter, Location loc,
                       llvm::StringMap<memref::AllocOp> &symbolMap) {
  return createConstantInt(rewriter, loc, value, 64);
}

// BoolNode
Value BoolNode::codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<memref::AllocOp> &symbolMap) {
  return createConstantInt(rewriter, loc, value, 1);
}

// VarNode
Value VarNode::codegen(PatternRewriter &rewriter, Location loc,
                       llvm::StringMap<memref::AllocOp> &symbolMap) {
  allocSymbol(rewriter, loc, name, symbolMap);
  return createLoad(rewriter, loc, symbolMap[name], {});
}

// AssignNode
Value AssignNode::codegen(PatternRewriter &rewriter, Location loc,
                          llvm::StringMap<memref::AllocOp> &symbolMap) {
  allocSymbol(rewriter, loc, variable->name, symbolMap);
  Value eVal = expr->codegen(rewriter, loc, symbolMap);
  createStore(rewriter, loc, eVal, symbolMap[variable->name], {});
  return nullptr;
}

// UnOpNode
Value UnOpNode::codegen(PatternRewriter &rewriter, Location loc,
                        llvm::StringMap<memref::AllocOp> &symbolMap) {
  // TODO: Code generation logic
  Value eVal = expr->codegen(rewriter, loc, symbolMap);
  return eVal;
}

// BinOpNode
Value BinOpNode::codegen(PatternRewriter &rewriter, Location loc,
                         llvm::StringMap<memref::AllocOp> &symbolMap) {
  // TODO: Code generation logic
  Value lVal = left->codegen(rewriter, loc, symbolMap);
  Value rVal = right->codegen(rewriter, loc, symbolMap);
  return lVal;
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

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

// stmt ::= assignment | log_or_expr
ASTNode *SymbolicParser::stmt() {
  ASTNode *assignNode = assignment();

  if (assignNode != nullptr)
    return assignNode;

  return log_or_expr();
}

// assignment ::= IDENT ASSIGN log_or_expr
ASTNode *SymbolicParser::assignment() {
  if (pos + 2 >= tokens.size() || tokens[pos].type != TokenType::IDENT ||
      tokens[pos + 1].type != TokenType::ASSIGN)
    return nullptr;

  std::string varName = tokens[pos].value;
  pos += 2;
  ASTNode *orNode = log_or_expr();

  if (orNode == nullptr)
    return nullptr;

  VarNode *varNode = new VarNode(varName);
  AssignNode *assignNode = new AssignNode(varNode, orNode);
  return assignNode;
}

// log_or_expr ::= log_and_expr ( LOG_OR log_and_expr )*
ASTNode *SymbolicParser::log_or_expr() {
  ASTNode *leftNode = log_and_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::LOG_OR) {
    pos++;
    ASTNode *rightNode = log_and_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::LOG_OR, rightNode);
  }

  return leftNode;
}

// log_and_expr ::= eq_expr ( LOG_AND eq_expr )*
ASTNode *SymbolicParser::log_and_expr() {
  ASTNode *leftNode = eq_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::LOG_AND) {
    pos++;
    ASTNode *rightNode = eq_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::LOG_AND, rightNode);
  }

  return leftNode;
}

// eq_expr ::= rel_expr ( ( EQ | NE ) rel_expr )*
ASTNode *SymbolicParser::eq_expr() {
  ASTNode *leftNode = rel_expr();

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
    ASTNode *rightNode = rel_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, binOp, rightNode);
  }

  return leftNode;
}

// rel_expr ::= shift_expr ( ( LT | LE | GT | GE ) shift_expr )*
ASTNode *SymbolicParser::rel_expr() {
  ASTNode *leftNode = shift_expr();

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
    ASTNode *rightNode = shift_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, binOp, rightNode);
  }

  return leftNode;
}

// shift_expr ::= bit_or_expr ( ( LSHIFT | RSHIFT ) bit_or_expr )*
ASTNode *SymbolicParser::shift_expr() {
  ASTNode *leftNode = bit_or_expr();

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
    ASTNode *rightNode = bit_or_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, binOp, rightNode);
  }

  return leftNode;
}

// bit_or_expr ::= bit_xor_expr ( BIT_OR bit_xor_expr )*
ASTNode *SymbolicParser::bit_or_expr() {
  ASTNode *leftNode = bit_xor_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_OR) {
    pos++;
    ASTNode *rightNode = bit_xor_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::BIT_OR, rightNode);
  }

  return leftNode;
}

// bit_xor_expr ::= bit_and_expr ( BIT_XOR bit_and_expr )*
ASTNode *SymbolicParser::bit_xor_expr() {
  ASTNode *leftNode = bit_and_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_XOR) {
    pos++;
    ASTNode *rightNode = bit_and_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::BIT_XOR, rightNode);
  }

  return leftNode;
}

// bit_and_expr ::= add_expr ( BIT_AND add_expr )*
ASTNode *SymbolicParser::bit_and_expr() {
  ASTNode *leftNode = add_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::BIT_AND) {
    pos++;
    ASTNode *rightNode = add_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::BIT_AND, rightNode);
  }

  return leftNode;
}

// add_expr ::= mul_expr ( ( ADD | SUB ) mul_expr )*
ASTNode *SymbolicParser::add_expr() {
  ASTNode *leftNode = mul_expr();

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
    ASTNode *rightNode = mul_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, binOp, rightNode);
  }

  return leftNode;
}

// mul_expr ::= exp_expr ( ( MUL | DIV | FLOORDIV | MOD ) exp_expr )*
ASTNode *SymbolicParser::mul_expr() {
  ASTNode *leftNode = exp_expr();

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
    ASTNode *rightNode = exp_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, binOp, rightNode);
  }

  return leftNode;
}

// exp_expr ::= unary_expr ( EXP unary_expr )*
ASTNode *SymbolicParser::exp_expr() {
  ASTNode *leftNode = unary_expr();

  if (leftNode == nullptr)
    return nullptr;

  while (pos + 1 < tokens.size() && tokens[pos].type == TokenType::EXP) {
    pos++;
    ASTNode *rightNode = unary_expr();

    if (rightNode == nullptr) {
      delete leftNode;
      return nullptr;
    }

    leftNode = new BinOpNode(leftNode, BinOpNode::BinOp::EXP, rightNode);
  }

  return leftNode;
}

// unary_expr ::= ( ADD | SUB | LOG_NOT | BIT_NOT )? factor
ASTNode *SymbolicParser::unary_expr() {
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
  ASTNode *node = factor();
  if (node == nullptr)
    return nullptr;
  UnOpNode *unopNode = new UnOpNode(unop, node);
  return unopNode;
}

// factor ::= LPAREN log_or_expr RPAREN | const_expr | IDENT
ASTNode *SymbolicParser::factor() {
  if (pos >= tokens.size())
    return nullptr;

  if (tokens[pos].type == TokenType::LPAREN) {
    pos++;
    ASTNode *expr = log_or_expr();
    if (expr == nullptr)
      return nullptr;

    if (pos >= tokens.size() || tokens[pos].type != TokenType::RPAREN) {
      delete expr;
      return nullptr;
    }

    pos++;
    return expr;
  }

  ASTNode *constExpr = const_expr();
  if (constExpr != nullptr)
    return constExpr;

  if (tokens[pos].type == TokenType::IDENT) {
    VarNode *varNode = new VarNode(tokens[pos].value);
    pos++;
    return varNode;
  }

  return nullptr;
}

// const_expr ::= bool_const | INT_CONST
ASTNode *SymbolicParser::const_expr() {
  if (pos >= tokens.size())
    return nullptr;

  ASTNode *boolExpr = bool_const();
  if (boolExpr != nullptr)
    return boolExpr;

  if (tokens[pos].type == TokenType::INT_CONST) {
    IntNode *intNode = new IntNode(std::stoi(tokens[pos].value));
    pos++;
    return intNode;
  }

  return nullptr;
}

// bool_const ::= TRUE | FALSE
ASTNode *SymbolicParser::bool_const() {
  if (pos >= tokens.size())
    return nullptr;

  if (tokens[pos].type == TokenType::TRUE) {
    pos++;
    BoolNode *boolNode = new BoolNode(true);
    return boolNode;
  }

  if (tokens[pos].type == TokenType::FALSE) {
    pos++;
    BoolNode *boolNode = new BoolNode(false);
    return boolNode;
  }

  return nullptr;
}

ASTNode *SymbolicParser::parse(StringRef input) {
  Optional<SmallVector<Token>> tokens = tokenize(input);
  if (!tokens.has_value())
    return nullptr;

  this->tokens = tokens.value();
  this->pos = 0;
  return stmt();
}

} // namespace mlir::sdfg::conversion
