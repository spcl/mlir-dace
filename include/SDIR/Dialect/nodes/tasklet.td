#ifndef SDIR_TaskletNode
#define SDIR_TaskletNode

def SDIR_TaskletNode : SDIR_Op<"tasklet", [
    HasParent<"StateNode">,
    AffineScope, 
    AutomaticAllocationScope, 
    CallableOpInterface, 
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
        TypeAttr:$type, 
        OptionalAttr<StrAttr>:$sym_visibility
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
        CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];

    let extraClassDeclaration = [{
        static TaskletNode create(PatternRewriter &rewriter, Location location,  FunctionType type);
        static TaskletNode create(Location location, StringRef name,
                                  FunctionType type, 
                                  ArrayRef<NamedAttribute> attrs = {});
        static TaskletNode create(Location location, StringRef name, 
                                  FunctionType type, 
                                  Operation::dialect_attr_range attrs);
        static TaskletNode create(Location location, StringRef name, 
                                  FunctionType type,
                                  ArrayRef<NamedAttribute> attrs,
                                  ArrayRef<DictionaryAttr> argAttrs);

        TaskletNode clone(BlockAndValueMapping &mapper);
        TaskletNode clone();

        void cloneInto(TaskletNode dest, BlockAndValueMapping &mapper);
        void setID(unsigned id);

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getType().getResults(); }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }

        private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<TaskletNode>;

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

#endif // SDIR_TaskletNode
