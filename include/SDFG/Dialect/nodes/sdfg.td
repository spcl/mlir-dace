#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

def SDFG_SDFGNode : SDFG_Op<"sdfg", [
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
        sdfg.sdfg{entry=@state_0} {
            sdfg.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins 
        I32Attr:$ID,
        SymbolNameAttr:$sym_name, 
        FlatSymbolRefAttr:$entry,
        I32Attr:$num_args,
        Variadic<AnyType>:$operands
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        static SDFGNode create(PatternRewriter &rewriter, Location location, FunctionType type);
        static SDFGNode create(PatternRewriter &rewriter, Location location, FunctionType type, StringRef name);
        unsigned getIndexOfState(StateNode &node);
        StateNode getStateByIndex(unsigned idx);
        StateNode getStateBySymRef(StringRef symRef);
        bool isNested();
        void setID(unsigned id);
    }];
}

#endif // SDFG_SDFGNode
