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
        sdir.map ($i, $j) = (0, 0) to (2, 2) step (1, 1) {
            ...
            %a = sdir.load %A[$i, $j] : !sdir.memlet<12x34xi32>
            ...
        } 
        ```
    }];

    let arguments = (ins 
        /*I32ElementsAttr:$lowerBounds,
        I32ElementsAttr:$upperBounds,*/
        I64ArrayAttr:$lowerBounds,
        I64ArrayAttr:$upperBounds,
        I64ArrayAttr:$steps
    );

    let regions = (region SizedRegion<1>:$region);
    let parser = [{ return ::parseMapNode(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    let verifier = [{ return :: verify(*this); }];
}

#endif // SDIR_MapNode
