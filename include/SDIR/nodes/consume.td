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
        Describes a consume scope where the region has access to the popped element as well as the .
        For example:

        ```mlir
        sdir.consume{num_pes=$P, condition=@empty} %A : !sdir.stream<i32> {
            ^bb0(%p: i32, %a: i32)
                %c = sdir.call @add_one(%a) : i32 -> i32
                sdir.store(wcr="add") %c, %C[] : i32 -> !sdir.memlet<i32>
            ...
        } 
        ```
    }];

    let arguments = (ins
        OptionalAttr<APIntAttr>:$num_pes, 
        OptionalAttr<FlatSymbolRefAttr>:$condition,
        SDIR_StreamType:$stream
    );

    let regions = (region SizedRegion<1>:$region);
    let parser = [{ return ::parseConsumeNode(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif // SDIR_ConsumeNode
