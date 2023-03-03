#ifndef SDFG_ConsumeNode
#define SDFG_ConsumeNode

def SDFG_ConsumeNode : SDFG_Op<"consume", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "Consume scope";
    let description = [{
        Describes a consume scope where the region has access to the popped 
        element as well as the processing element.
        For example:

        ```mlir
        sdfg.consume{num_pes=5} (%a : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
                %c = sdfg.call @add_one(%a) : i32 -> i32
                sdfg.store(wcr="add") %c, %C[] : i32 -> !sdfg.memlet<i32>
            ...
        } 
        ```
    }];

    let arguments = (ins
        I32Attr:$entryID,
        I32Attr:$exitID,
        OptionalAttr<APIntAttr>:$num_pes, 
        OptionalAttr<FlatSymbolRefAttr>:$condition,
        SDFG_StreamType:$stream
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        BlockArgument pe();
        BlockArgument elem();
    }];
}

#endif // SDFG_ConsumeNode
