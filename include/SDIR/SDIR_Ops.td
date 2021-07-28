#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"
include "SDIR/SDIR_Types.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

//===----------------------------------------------------------------------===//
// State Region
//===----------------------------------------------------------------------===//

def SDIR_StateOp : SDIR_Op<"state", [
    SingleBlock,
    NoTerminator
]> {
    let summary = "State region";
    let description = [{
        Describes a state where the subgraphs are placed in the region. 

        Example:
        sdir.state @state_0{

        } 
    }];

    let arguments = (ins SymbolNameAttr:$name);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{attr-dict $name $region}];
}

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

def SDIR_TaskletOp : SDIR_Op<"tasklet", [
    AffineScope, 
    AutomaticAllocationScope, 
    CallableOpInterface, 
    FunctionLike,
    IsolatedFromAbove, 
    Symbol
]> {
    let summary = "Tasklet";
    let description = [{
        Describes a tasklet as a pure function. For example:

        ```mlir
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = addi %a, %b, : i32
            sdir.return %c
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type, OptionalAttr<StrAttr>:$sym_visibility);
    let regions = (region AnyRegion:$body);
 
    let parser = [{
        auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                                function_like_impl::VariadicFlag, std::string &) {
            return builder.getFunctionType(argTypes, results);
        };

        return function_like_impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false, buildFuncType);
    }];

    let printer =[{
        FunctionType fnType = getType();
        function_like_impl::printFunctionLikeOp(p, *this, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
    }];

    let verifier = [{ 
        if (isExternal())
            return success();

        // Verify that the argument list of the function and the arg list of the entry
        // block line up.  The trait already verified that the number of arguments is
        // the same between the signature and the block.
        auto fnInputTypes = getType().getInputs();
        Block &entryBlock = front();
        for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
            if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
            return emitOpError("type of entry block argument #")
                    << i << '(' << entryBlock.getArgument(i).getType()
                    << ") must match the type of the corresponding argument in "
                    << "function signature(" << fnInputTypes[i] << ')';

        return success();
    }];

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
        CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getType().getResults(); }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }


        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<FuncOp>;

        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getType().getInputs().size(); }

        /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getType().getResults().size(); }

        LogicalResult verifyType() {
            auto type = getTypeAttr().getValue();
            if (!type.isa<FunctionType>())
                return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
            return success();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Return
//===----------------------------------------------------------------------===//

def SDIR_ReturnOp : SDIR_Op<"return", [
    Terminator
]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
    func @foo() -> tensor<2xf64> {
        ...
        sdir.return %0 : tensor<2xf64>
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
}

#endif // SDIR_OPS