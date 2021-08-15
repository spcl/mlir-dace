#ifndef SDIR_LibCallOp
#define SDIR_LibCallOp

def SDIR_LibCallOp : SDIR_Op<"libcall", [
    HasParent<"StateOp">,
    CallOpInterface
]> {
  let summary = "library call operation";
  let description = [{
    The `libcall` operation represents a direct call to a library function. 
    Example:

    ```mlir
    %2 = sdir.libcall "dace.libraries.blas.nodes.Gemm" (%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{ 
    attr-dict $callee `(` $operands `)` `:` functional-type($operands, results) 
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
}

#endif // SDIR_LibCallOp
