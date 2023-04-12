#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

def SDFG_SDFGNode : SDFG_Op<"sdfg", [
    ParentOneOf<["ModuleOp"]>,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
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
        OptionalAttr<FlatSymbolRefAttr>:$entry,
        I32Attr:$num_args
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        static SDFGNode create(PatternRewriter &rewriter, Location location, unsigned num_args, TypeRange args);
        static SDFGNode create(PatternRewriter &rewriter, Location location);
        StateNode getStateBySymRef(StringRef symRef);
        StateNode getFirstState();
        StateNode getEntryState();
        Block::BlockArgListType getArgs();
        TypeRange getArgTypes();
        Block::BlockArgListType getResults();
        TypeRange getResultTypes();
    }];
}

def SDFG_NestedSDFGNode : SDFG_Op<"nested_sdfg", [
    ParentOneOf<["StateNode","MapNode","ConsumeNode"]>,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "Nested SDFG region";
    let description = [{
        Describes a nested SDFG where the states are placed in the region. 
        For example:

        ```mlir
        sdfg.nested_sdfg{entry=@state_0} {
            sdfg.edge{assign=["i = 1"]} @state_0 -> @state_1
            ...
        }
        ```
    }];

    let arguments = (ins 
        I32Attr:$ID,
        OptionalAttr<FlatSymbolRefAttr>:$entry,
        I32Attr:$num_args,
        Variadic<AnyType>:$operands
    );

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        static NestedSDFGNode create(PatternRewriter &rewriter, Location location, unsigned num_args, ValueRange args);
        static NestedSDFGNode create(PatternRewriter &rewriter, Location location);
        StateNode getStateBySymRef(StringRef symRef);
        StateNode getFirstState();
        StateNode getEntryState();
        ValueRange getArgs();
        ValueRange getResults();
    }];
}

#endif // SDFG_SDFGNode
