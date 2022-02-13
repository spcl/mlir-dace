#ifndef SDFG_TaskletNode
#define SDFG_TaskletNode

def SDFG_TaskletNode : SDFG_Op<"tasklet", [
    ParentOneOf<["StateNode","MapNode","ConsumeNode"]>,
    AffineScope, 
    AutomaticAllocationScope, 
    IsolatedFromAbove, 
    Symbol
]> {
    let summary = "Tasklet";
    let description = [{
        Describes a tasklet as a pure function. For example:

        ```mlir
        sdfg.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b, : i32
            sdfg.return %c
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
        static TaskletNode create(PatternRewriter &rewriter, Location location, 
                                  ValueRange operands, TypeRange results);
        static TaskletNode create(Location location, StringRef name,
                                  ValueRange operands, TypeRange results);

        void setID(unsigned id);
        std::string getInputName(unsigned idx);
        std::string getOutputName(unsigned idx);
    }];
}

#endif // SDFG_TaskletNode
