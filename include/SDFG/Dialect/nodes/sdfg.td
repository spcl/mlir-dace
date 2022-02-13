#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

def SDFG_SDFGNode : SDFG_Op<"sdfg", [
    NoTerminator,
    IsolatedFromAbove,
    FunctionLike,
    CallableOpInterface,
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
        TypeAttr:$type
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
        
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        Region *getCallableRegion(){ return isExternal() ? nullptr : &getRegion(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults(){ return getType().getResults(); }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }

        private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<SDFGNode>;

        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getType().getInputs().size(); }

        /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getType().getResults().size(); }

        LogicalResult verifyType() {
            auto type = getTypeAttr().getValue();
            if (!type.isa<FunctionType>())
                return emitOpError("requires '" + getTypeAttrName() 
                                    + "' attribute of function type");
            return success();
        }
    }];
}

#endif // SDIR_SDFGNode
