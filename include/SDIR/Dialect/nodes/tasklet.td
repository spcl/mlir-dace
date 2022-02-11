#ifndef SDIR_TaskletNode
#define SDIR_TaskletNode

def SDIR_TaskletNode : SDIR_Op<"tasklet", [
    HasParent<"StateNode">,
    AffineScope, 
    AutomaticAllocationScope, 
    FunctionLike,
    IsolatedFromAbove, 
    Symbol
]> {
    let summary = "Tasklet";
    let description = [{
        Describes a tasklet as a pure function. For example:

        ```mlir
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b, : i32
            sdir.return %c
        }
        ```
    }];

    let arguments = (ins
        I32Attr:$ID,
        SymbolNameAttr:$sym_name, 
        TypeAttr:$signature
    );
    let results = (outs Variadic<AnyType>:$res);
    let regions = (region AnyRegion:$body);


    let extraClassDeclaration = [{
        static TaskletNode create(PatternRewriter &rewriter, Location location,  FunctionType type);
        static TaskletNode create(Location location,  StringRef name, FunctionType type);

        void setID(unsigned id);
        std::string getInputName(unsigned idx);
        std::string getOutputName(unsigned idx);

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }

        private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<TaskletNode>;

        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return FunctionLike::getType().getInputs().size(); }

        /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return FunctionLike::getType().getResults().size(); }

        LogicalResult verifyType() {
            auto type = getTypeAttr().getValue();
            if (!type.isa<FunctionType>())
                return emitOpError("requires '" + getTypeAttrName() 
                                    + "' attribute of function type");
            return success();
        }
    }];
}

#endif // SDIR_TaskletNode
