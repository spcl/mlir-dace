#ifndef SDIR_TaskletOps
#define SDIR_TaskletOps

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def SDIR_ReturnOp : SDIR_Op<"return", [
    HasParent<"TaskletNode">,
    Terminator
]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. 
        For example:

        ```mlir
        func @foo() -> tensor<2xf64> {
            ...
            sdir.return %0 : tensor<2xf64>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$input);

    let extraClassDeclaration = [{
        static sdir::ReturnOp create(PatternRewriter &rewriter, Location loc, mlir::ValueRange input);
        static sdir::ReturnOp create(Location loc, mlir::ValueRange input);
    }];
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

def SDIR_LibCallOp : SDIR_Op<"libcall", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    CallOpInterface
]> {
    let summary = "library call operation";
    let description = [{
        The `libcall` operation represents a direct call to a library function. 
        For example:

        ```mlir
        %2 = sdir.libcall "dace.libraries.blas.nodes.Dot" (%0, %1) : (f32, f32) -> f32
        ```
    }];

    let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>);

    let extraClassDeclaration = [{
        operand_range getArgOperands() {
            return {operand_begin(), operand_end()};
        }

        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }
    }];
}

#endif // SDIR_TaskletOps
