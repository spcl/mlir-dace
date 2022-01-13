#ifndef SDIR_GetAccessOp
#define SDIR_GetAccessOp

def SDIR_GetAccessOp : SDIR_Op<"get_access", [
    HasParent<"StateNode">
]> {
    let summary = "Array access operation";
    let description = [{
        Allows a state to access the specified array.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %a = sdir.get_access %A : !sdir.memlet<i32>
            ...
        }
        ```
    }];

    let arguments = (ins 
        I32Attr:$ID,
        AnyTypeOf<[SDIR_ArrayType, SDIR_StreamArrayType]>:$arr
    );
    let results = (outs AnyTypeOf<[SDIR_MemletType, SDIR_StreamType]>:$res);

    let extraClassDeclaration = [{
        static GetAccessOp create(PatternRewriter &rewriter, Location loc, Type t, Value arr);
        Type getAllocType();
        ArrayRef<StringAttr> getSymbols();
        ArrayRef<int64_t> getIntegers();
        ArrayRef<bool> getShape();
        SDFGNode getParentSDFG();
        std::string getName();
        void setID(unsigned id);
    }];
}

#endif // SDIR_GetAccessOp
