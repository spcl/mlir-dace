// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for JSON emitter in SDFG translation.

#ifndef SDFG_JsonEmitter_H
#define SDFG_JsonEmitter_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::sdfg::emitter {

struct JsonEmitter {
  explicit JsonEmitter(raw_ostream &os);

  /// Returns a reference to the output stream. Avoid writing directly to the
  /// output stream if possible.
  raw_ostream &ostream() { return os; };
  /// Returns the current indentation level.
  unsigned getIndentation() { return indentation; };
  /// Checks for errors (open objects/lists) and adds trailing newline. Returns
  /// a LogicalResult indicating success or failure.
  LogicalResult finish();

  /// Increases the indentation level.
  void indent();
  /// Decreases the indentation level.
  void unindent();
  /// Starts a new line in the output stream.
  void newLine();
  /// Prints a literal string to the output stream.
  void printLiteral(StringRef str);
  /// Prints a string to the output stream, surrounding it with quotation marks.
  void printString(StringRef str);
  /// Prints an integer to the output stream, surrounding it with quotation
  /// marks.
  void printInt(int i);

  /// Starts a new JSON object.
  void startObject();
  /// Starts a new named (keyed) JSON object.
  void startNamedObject(StringRef name);
  /// Ends the current JSON object.
  void endObject();

  /// Starts a new named JSON list.
  void startNamedList(StringRef name);
  /// Ends the current JSON list.
  void endList();

  /// Starts a new entry in the current JSON object or list.
  void startEntry();
  /// Prints a key-value pair to the output stream. If desired, turns the value
  /// into string.
  void printKVPair(StringRef key, StringRef val, bool stringify = true);
  /// Prints a key-value pair to the output stream. If desired, turns the value
  /// into string.
  void printKVPair(StringRef key, int val, bool stringify = true);
  /// Prints a key-value pair to the output stream. If desired, turns the value
  /// into string.
  void printKVPair(StringRef key, Attribute val, bool stringify = true);

  /// Prints a list of NamedAttributes as key-value pairs.
  void printAttributes(ArrayRef<NamedAttribute> arr,
                       ArrayRef<StringRef> elidedAttrs = {});

private:
  /// The output stream.
  raw_ostream &os;
  /// The current indentation level.
  unsigned indentation;
  /// Flag indicating whether the current entry is the first in its parent
  /// object or list.
  bool firstEntry;
  /// Flag indicating whether the current line is empty.
  bool emptyLine;
  /// Enum class to represent the type of the current JSON symbol.
  enum class SYM {
    /// "{" or "}"
    BRACE,
    /// "[" or "]"
    SQUARE
  };

  /// Stack to keep track of the opened JSON symbols.
  SmallVector<SYM> symStack;
  /// Tries to pop a symbol from the symStack, checking for matching symbols.
  void tryPop(SYM sym);

  /// Flag indicating whether there was an error during printing.
  bool error;
};

} // namespace mlir::sdfg::emitter

#endif // SDFG_JsonEmitter_H
