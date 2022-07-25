#ifndef SDFG_TaskletNode
#define SDFG_TaskletNode

def SDFG_TaskletNode : SDFG_Op<"tasklet", [
    ParentOneOf<["StateNode","MapNode","ConsumeNode"]>,
    AffineScope, 
    AutomaticAllocationScope, 
    IsolatedFromAbove
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
        Variadic<AnyType>:$operands
    );

    let results = (outs Variadic<AnyType>);
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        static TaskletNode create(PatternRewriter &rewriter, Location location, 
                                  ValueRange operands, TypeRange results);
        static TaskletNode create(Location location, ValueRange operands, 
                                  TypeRange results);

        std::string getInputName(unsigned idx);
        std::string getOutputName(unsigned idx);
    }];
}

#endif // SDFG_TaskletNode
