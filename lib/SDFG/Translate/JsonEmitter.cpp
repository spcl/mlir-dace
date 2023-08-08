// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Translate/JsonEmitter.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace sdfg;
using namespace emitter;

/// Creates a new JSON emitter.
JsonEmitter::JsonEmitter(raw_ostream &os) : os(os) {
  indentation = 0;
  error = false;
  firstEntry = true;
  emptyLine = true;
  symStack.clear();
}

/// Checks for errors (open objects/lists) and adds trailing newline. Returns
/// a LogicalResult indicating success or failure.
LogicalResult JsonEmitter::finish() {
  while (!symStack.empty()) {
    SYM sym = symStack.pop_back_val();
    newLine();
    unindent();
    if (sym == SYM::BRACE)
      printLiteral("}");
    if (sym == SYM::SQUARE)
      printLiteral("]");
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Auto-inserted closing clause");
    os.resetColor();
    error = true;
  }
  newLine(); // Makes sure to have a trailing newline
  return failure(error);
}

/// Increases the indentation level.
void JsonEmitter::indent() { indentation += 2; }
/// Decreases the indentation level.
void JsonEmitter::unindent() {
  indentation = indentation >= 2 ? indentation - 2 : 0;
}

/// Starts a new line in the output stream.
void JsonEmitter::newLine() {
  if (emptyLine)
    return;
  os << "\n";
  emptyLine = true;
}

/// Prints a literal string to the output stream.
void JsonEmitter::printLiteral(StringRef str) {
  if (emptyLine)
    os.indent(indentation);
  os << str;
  emptyLine = false;
}

/// Prints a string to the output stream, surrounding it with quotation marks.
void JsonEmitter::printString(StringRef str) {
  printLiteral("\"");
  printLiteral(str);
  printLiteral("\"");
}

/// Prints an integer to the output stream, surrounding it with quotation
/// marks.
void JsonEmitter::printInt(int i) {
  printLiteral("\"");
  os << i;
  printLiteral("\"");
}

/// Starts a new JSON object.
void JsonEmitter::startObject() {
  startEntry();
  printLiteral("{");
  if (!symStack.empty() && symStack.back() == SYM::BRACE) {
    // Need a key inside objects
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Started object without a key");
    os.resetColor();
    error = true;
  }
  symStack.push_back(SYM::BRACE);
  indent();
  newLine();
  firstEntry = true;
}

/// Starts a new named (keyed) JSON object.
void JsonEmitter::startNamedObject(StringRef name) {
  startEntry();
  printString(name);
  printLiteral(": ");
  printLiteral("{");
  if (symStack.empty() || symStack.back() == SYM::SQUARE) {
    // Can't have keyed values as root object or in a list
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Started keyed object in a list");
    os.resetColor();
    error = true;
  }
  symStack.push_back(SYM::BRACE);
  indent();
  newLine();
  firstEntry = true;
}

/// Ends the current JSON object.
void JsonEmitter::endObject() {
  newLine();
  unindent();
  tryPop(SYM::BRACE);
  printLiteral("}");
  firstEntry = false;
}

/// Starts a new named JSON list.
void JsonEmitter::startNamedList(StringRef name) {
  startEntry();
  printString(name);
  printLiteral(": ");
  printLiteral("[");
  if (!symStack.empty() && symStack.back() == SYM::SQUARE) {
    // Can't have keyed values in a list
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Started keyed list in a list");
    os.resetColor();
    error = true;
  }
  symStack.push_back(SYM::SQUARE);
  indent();
  newLine();
  firstEntry = true;
}

/// Ends the current JSON list.
void JsonEmitter::endList() {
  newLine();
  unindent();
  tryPop(SYM::SQUARE);
  printLiteral("]");
  firstEntry = false;
}

/// Starts a new entry in the current JSON object or list.
void JsonEmitter::startEntry() {
  if (!firstEntry)
    printLiteral(",");
  firstEntry = false;
  newLine();
}

/// Prints a key-value pair to the output stream. If desired, turns the value
/// into string.
void JsonEmitter::printKVPair(StringRef key, StringRef val, bool stringify) {
  startEntry();
  printString(key);
  printLiteral(": ");
  if (stringify)
    printString(val);
  else
    printLiteral(val);
}

/// Prints a key-value pair to the output stream. If desired, turns the value
/// into string.
void JsonEmitter::printKVPair(StringRef key, int val, bool stringify) {
  startEntry();
  printString(key);
  printLiteral(": ");
  if (stringify)
    printInt(val);
  else
    os << val;
}

/// Prints a key-value pair to the output stream. If desired, turns the value
/// into string.
void JsonEmitter::printKVPair(StringRef key, Attribute val, bool stringify) {
  startEntry();
  printString(key);
  printLiteral(": ");
  if (StringAttr strAttr = val.dyn_cast<StringAttr>()) {
    strAttr.print(os);
  } else {
    if (stringify)
      printLiteral("\"");
    val.print(os);
    if (stringify)
      printLiteral("\"");
  }
}

/// Prints a list of NamedAttributes as key-value pairs.
void JsonEmitter::printAttributes(ArrayRef<NamedAttribute> arr,
                                  ArrayRef<StringRef> elidedAttrs) {

  llvm::SmallDenseSet<StringRef> elidedAttrsSet(elidedAttrs.begin(),
                                                elidedAttrs.end());

  for (NamedAttribute attr : arr) {
    if (elidedAttrsSet.contains(attr.getName().strref()))
      continue;
    printKVPair(attr.getName().strref(), attr.getValue());
  }
}

/// Tries to pop a symbol from the symStack, checking for matching symbols.
void JsonEmitter::tryPop(SYM sym) {
  if (symStack.empty()) {
    if (sym == SYM::BRACE)
      printLiteral("{");
    if (sym == SYM::SQUARE)
      printLiteral("[");
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Auto-inserted opening clause");
    os.resetColor();
    newLine();
    indent();
    error = true;
  } else if (symStack.back() != sym) {
    if (sym == SYM::BRACE)
      printLiteral("]");
    if (sym == SYM::SQUARE)
      printLiteral("}");
    os.changeColor(os.RED, /*Bold=*/true);
    printLiteral(" <<<<<<<<<<<< Auto-inserted closing clause");
    os.resetColor();
    newLine();
    unindent();
    error = true;
    symStack.pop_back();
  } else {
    symStack.pop_back();
  }
}
