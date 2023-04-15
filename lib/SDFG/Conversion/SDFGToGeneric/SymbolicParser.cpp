#include "SDFG/Conversion/SDFGToGeneric/SymbolicParser.h"

using namespace mlir;
using namespace sdfg::conversion;

// Implements a simple recursive descent parser for symbolic expressions

/*
expr ::= "(" expr ")"
       | unop expr
       | expr binop expr
       | assignment
       | variable
       | constant

constant ::= digit, {digit}
variable ::= identifier
assignment ::= identifier, ":", expr
unop ::= "-" | "~" | "not"
binop ::= "+" | "-" | "*" | "/" | "//" | "%" | "&" | "|" | "^" | "<<" | ">>"
        | "**" | "and" | "or" | "<" | "<=" | ">" | ">=" | "==" | "!="

identifier ::= (letter | "_"), {letter | "_" | digit}

digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
letter ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k"
         | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v"
         | "w" | "x" | "y" | "z" | "A" | "B" | "C" | "D" | "E" | "F" | "G"
         | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R"
         | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
*/
namespace mlir::sdfg::conversion {

void skip_whitespace(StringRef input, unsigned &pos) {
  while (pos < input.size() && isspace(input[pos]))
    ++pos;
}

std::unique_ptr<Node> parse_parenthesized(StringRef input, unsigned &pos) {
  unsigned prev_pos = pos;
  skip_whitespace(input, pos);

  if (input[pos] == '(') {
    ++pos;
    std::unique_ptr<Node> node = SymbolicParser::parse(input, pos);

    if (!node) {
      pos = prev_pos;
      return nullptr;
    }

    skip_whitespace(input, pos);

    if (input[pos] == ')') {
      ++pos;
      return node;
    }
  }

  pos = prev_pos;
  return nullptr;
}

class VariableNode : public Node {
private:
  std::string name;

public:
  VariableNode(const std::string &name) : name(name) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for variables
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    variables.push_back(name);
  }

  static std::unique_ptr<VariableNode> parse(StringRef input, unsigned &pos) {
    unsigned prev_pos = pos;
    skip_whitespace(input, pos);

    if (!isalpha(input[pos]) && input[pos] != '_') {
      pos = prev_pos;
      return nullptr;
    }

    std::string name;
    while (pos < input.size() && (isalnum(input[pos]) || input[pos] == '_'))
      name += input[pos++];

    return std::make_unique<VariableNode>(name);
  }
};

class ConstantNode : public Node {
private:
  int value;

public:
  ConstantNode(int value) : value(value) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for constants
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {}

  static std::unique_ptr<ConstantNode> parse(StringRef input, unsigned &pos) {
    unsigned prev_pos = pos;
    skip_whitespace(input, pos);

    if (!isdigit(input[pos])) {
      pos = prev_pos;
      return nullptr;
    }

    int value = 0;
    while (pos < input.size() && isdigit(input[pos]))
      value = value * 10 + (input[pos++] - '0');

    return std::make_unique<ConstantNode>(value);
  }
};

class AssignmentNode : public Node {
private:
  std::unique_ptr<VariableNode> variable;
  std::unique_ptr<Node> expr;

public:
  AssignmentNode(std::unique_ptr<VariableNode> variable,
                 std::unique_ptr<Node> expr)
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

  static std::unique_ptr<AssignmentNode> parse(StringRef input, unsigned &pos) {
    unsigned prev_pos = pos;
    std::unique_ptr<VariableNode> variable = VariableNode::parse(input, pos);

    if (!variable) {
      pos = prev_pos;
      return nullptr;
    }

    if (pos >= input.size() || input[pos] != ':') {
      pos = prev_pos;
      return nullptr;
    }

    pos++;
    std::unique_ptr<Node> expr = SymbolicParser::parse(input, pos);
    if (!expr) {
      pos = prev_pos;
      return nullptr;
    }

    return std::make_unique<AssignmentNode>(std::move(variable),
                                            std::move(expr));
  }
};

class UnOpNode : public Node {
private:
  std::string op;
  std::unique_ptr<Node> expr;

public:
  UnOpNode(std::string op, std::unique_ptr<Node> expr)
      : op(op), expr(std::move(expr)) {}

  mlir::Value codegen(mlir::PatternRewriter &rewriter,
                      mlir::Location loc) override {
    // TODO: Code generation logic for unary operators
    return nullptr;
  }

  void collect_variables(SmallVector<std::string> &variables) override {
    expr->collect_variables(variables);
  }

  static std::unique_ptr<UnOpNode> parse(StringRef input, unsigned &pos) {
    unsigned prev_pos = pos;
    skip_whitespace(input, pos);
    Optional<std::string> op = find_unop(input, pos);

    if (!op) {
      pos = prev_pos;
      return nullptr;
    }

    pos += op->length();
    std::unique_ptr<Node> expr = SymbolicParser::parse(input, pos);

    if (!expr) {
      pos = prev_pos;
      return nullptr;
    }

    return std::make_unique<UnOpNode>(*op, std::move(expr));
  }

  static Optional<std::string> find_unop(StringRef input, unsigned pos) {
    std::string binops[] = {"-", "~", "not"};

    for (std::string op : binops)
      if (input.substr(pos).startswith(op))
        return op;

    return std::nullopt;
  }
};

class BinOpNode : public Node {
private:
  std::unique_ptr<Node> left;
  std::string op;
  std::unique_ptr<Node> right;

public:
  BinOpNode(std::unique_ptr<Node> left, std::string op,
            std::unique_ptr<Node> right)
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

  static std::unique_ptr<BinOpNode> parse(StringRef input, unsigned &pos) {
    unsigned prev_pos = pos;
    std::unique_ptr<Node> left;

    // FIXME: Rewrite to respect operator precedence
    left = parse_parenthesized(input, pos);
    if (!left)
      left = UnOpNode::parse(input, pos);

    if (!left)
      left = VariableNode::parse(input, pos);

    if (!left)
      left = ConstantNode::parse(input, pos);

    if (!left) {
      pos = prev_pos;
      return nullptr;
    }

    skip_whitespace(input, pos);
    Optional<std::string> op = find_binop(input, pos);

    if (!op) {
      pos = prev_pos;
      return nullptr;
    }

    pos += op->length();
    std::unique_ptr<Node> right = SymbolicParser::parse(input, pos);

    if (!right) {
      pos = prev_pos;
      return nullptr;
    }

    return std::make_unique<BinOpNode>(std::move(left), *op, std::move(right));
  }

  static Optional<std::string> find_binop(StringRef input, unsigned pos) {
    std::string binops[] = {"+", "-",  "*",  "/",  "//", "%",   "&",
                            "|", "^",  "<<", ">>", "**", "and", "or",
                            "<", "<=", ">",  ">=", "==", "!="};

    for (std::string op : binops)
      if (input.substr(pos).startswith(op))
        return op;

    return std::nullopt;
  }
};

std::unique_ptr<Node> SymbolicParser::parse(StringRef input, unsigned pos) {
  printf("Parsing: %s\n", input.substr(pos).str().c_str());
  std::unique_ptr<Node> node;

  node = parse_parenthesized(input, pos);
  if (node)
    return node;

  node = UnOpNode::parse(input, pos);
  if (node)
    return node;

  node = BinOpNode::parse(input, pos);
  if (node)
    return node;

  node = AssignmentNode::parse(input, pos);
  if (node)
    return node;

  node = VariableNode::parse(input, pos);
  if (node)
    return node;

  node = ConstantNode::parse(input, pos);
  if (node)
    return node;

  return nullptr;
}

} // namespace mlir::sdfg::conversion
