module {
    func @bar() {
        %0 = constant 1 : i32
        %res = sdir.foo %0 : i32
        return
    }
}