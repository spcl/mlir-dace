#ifndef SDFG_MapNode
#define SDFG_MapNode

def SDFG_MapNode : SDFG_Op<"map", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> {
    let summary = "Map scope";
    let description = [{
        Describes a map where the region has access to the map symbol variables.
        For example:

        ```mlir
        sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            ...
            %a = sdfg.load %A[%i, %j] : !sdfg.memlet<12x34xi32>
            ...
        } 
        ```
    }];

    let arguments = (ins 
        I32Attr:$entryID,
        I32Attr:$exitID,
        Variadic<Index>:$ranges,
        ArrayAttr:$lowerBounds,
        ArrayAttr:$upperBounds,
        ArrayAttr:$steps
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        void setEntryID(unsigned id);
        void setExitID(unsigned id);
    }];
}

#endif // SDFG_MapNode