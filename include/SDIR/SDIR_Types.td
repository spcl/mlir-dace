#ifndef SDIR_Types
#define SDIR_Types

//===----------------------------------------------------------------------===//
// Base class for all SDIR types
//===----------------------------------------------------------------------===//

class SDIR_Type<string name, list<Trait> traits = []> : TypeDef<SDIR_Dialect, name, traits>{}

#endif // SDIR_Types