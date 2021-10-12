#ifndef SDIR_ConsumeNode
#define SDIR_ConsumeNode

def SDIR_ConsumeNode : SDIR_Op<"consume", [
    HasParent<"StateNode">,
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
        sdir.consume{num_pes=5} (%a : !sdir.stream<i32>) -> (pe: %p, elem: %e) {
                %c = sdir.call @add_one(%a) : i32 -> i32
                sdir.store(wcr="add") %c, %C[] : i32 -> !sdir.memlet<i32>
            ...
        } 
        ```
    }];

    let arguments = (ins
        I32Attr:$entryID,
        I32Attr:$exitID,
        OptionalAttr<APIntAttr>:$num_pes, 
        OptionalAttr<FlatSymbolRefAttr>:$condition,
        SDIR_StreamType:$stream
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        void setEntryID(unsigned id);
        void setExitID(unsigned id);
    }];
}

#endif // SDIR_ConsumeNode
