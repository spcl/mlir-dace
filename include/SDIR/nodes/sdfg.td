#ifndef SDIR_SDFGNode
#define SDIR_SDFGNode

def SDIR_SDFGNode : SDIR_Op<"sdfg", [
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "SDFG region";
    let description = [{
        Describes an SDFG where the states are placed in the region. 
        For example:

        ```mlir
        sdir.sdfg{entry=@state_0} {
            sdir.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$entry);
    let regions = (region SizedRegion<1>:$region);

    let parser = [{ return ::parseSDFGNode(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
}

#endif // SDIR_SDFGNode
