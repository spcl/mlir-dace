#ifndef SDIR_MapNode
#define SDIR_MapNode

def SDIR_MapNode : SDIR_Op<"map", [
    HasParent<"StateNode">,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> {
    let summary = "Map scope";
    let description = [{
        Describes a map where the region has access to the map symbol variables.
        For example:

        ```mlir
        sdir.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            ...
            %a = sdir.load %A[%i, %j] : !sdir.memlet<12x34xi32>
            ...
        } 
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$ranges,
        ArrayAttr:$lowerBounds,
        ArrayAttr:$upperBounds,
        ArrayAttr:$steps
    );

    let regions = (region SizedRegion<1>:$body);
}

#endif // SDIR_MapNode
