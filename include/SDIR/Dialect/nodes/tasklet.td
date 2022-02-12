#ifndef SDIR_TaskletNode
#define SDIR_TaskletNode

def SDIR_TaskletNode : SDIR_Op<"tasklet", [
    HasParent<"StateNode">,
    AffineScope, 
    AutomaticAllocationScope, 
    IsolatedFromAbove, 
    Symbol
]> {
    let summary = "Tasklet";
    let description = [{
        Describes a tasklet as a pure function. For example:

        ```mlir
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b, : i32
            sdir.return %c
        }
        ```
    }];

    let arguments = (ins
        I32Attr:$ID,
        SymbolNameAttr:$sym_name, 
        Variadic<AnyType>:$operands
    );
    
    let results = (outs Variadic<AnyType>);
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        static TaskletNode create(PatternRewriter &rewriter, Location location, FunctionType type);
        static TaskletNode create(Location location, StringRef name, FunctionType type);

        void setID(unsigned id);
        std::string getInputName(unsigned idx);
        std::string getOutputName(unsigned idx);
    }];
}

#endif // SDIR_TaskletNode
