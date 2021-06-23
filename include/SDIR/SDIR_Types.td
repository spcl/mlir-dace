#ifndef SDIR_Types
#define SDIR_Types

//===----------------------------------------------------------------------===//
// Base class for all SDIR types
//===----------------------------------------------------------------------===//

class SDIR_Type<string name, list<Trait> traits = []> : TypeDef<SDIR_Dialect, name, traits>{}

//===----------------------------------------------------------------------===//
// StateType
//===----------------------------------------------------------------------===//

/*def SDIR_StateType : SDIR_Type<"State"> {
  let mnemonic = "int";

  let summary = "A state machine element.";

  let description = [{
    A state machine element.
  }];

  let parameters = (ins "int":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}*/


#endif // SDIR_Types