#ifndef SDIR_CallOp
#define SDIR_CallOp

def SDIR_CallOp : SDIR_Op<"call", [
    HasParent<"StateOp">,
    CallOpInterface, 
    MemRefsNormalizable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a tasklet function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. 
    Example:

    ```mlir
    %2 = sdir.call @add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{ 
    attr-dict $callee `(` $operands `)` `:` functional-type($operands, results) 
  }];

  let builders = [
    OpBuilder<(ins "TaskletOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
  }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
}

#endif // SDIR_CallOp