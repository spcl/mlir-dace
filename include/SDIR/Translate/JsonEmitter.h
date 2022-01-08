#ifndef SDIR_JsonEmitter_H
#define SDIR_JsonEmitter_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::sdir::emitter {

struct JsonEmitter {
  explicit JsonEmitter(raw_ostream &os);

  // Avoid writing directly to the output stream if possible.
  raw_ostream &ostream() { return os; };
  unsigned getIndentation() { return indentation; };
  // Checks for errors (open objects/lists) and adds trailing newline
  LogicalResult finish();

  void indent();
  void unindent();
  void newLine();
  void printLiteral(StringRef str);
  void printString(StringRef str);
  void printInt(int i);

  void startObject();
  void startNamedObject(StringRef name);
  void endObject();

  void startNamedList(StringRef name);
  void endList();

  void startEntry();
  void printKVPair(StringRef key, StringRef val, bool stringify = true);
  void printKVPair(StringRef key, int val, bool stringify = true);
  void printKVPair(StringRef key, Attribute val, bool stringify = true);

  void printAttributes(ArrayRef<NamedAttribute> arr,
                       ArrayRef<StringRef> elidedAttrs = {});

private:
  // output stream
  raw_ostream &os;
  unsigned indentation;
  // Avoids printing commas for first entries (objects or lists)
  bool firstEntry;
  // Stores if the current line is empty or not
  bool emptyLine;
  // Used to check for proper closing of opened objects/lists
  enum class SYM {
    // "{" or "}"
    BRACE,
    // "[" or "]"
    SQUARE
  };

  SmallVector<SYM> symStack;
  void tryPop(SYM sym);

  // Tracks if there was an erronous printing
  bool error;
};

} // namespace mlir::sdir::emitter

#endif // SDIR_JsonEmitter_H
