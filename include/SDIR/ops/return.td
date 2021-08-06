#ifndef SDIR_ReturnOp
#define SDIR_ReturnOp

def SDIR_ReturnOp : SDIR_Op<"return", [
    HasParent<"TaskletOp">,
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

#endif // SDIR_ReturnOp