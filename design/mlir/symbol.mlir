sdir.alloc_symbol("N")

sdir.map $i = 0 to $N step 1 {
    %a = sdir.load %A[$i]
    %c = sdir.call @add_one(%a)
    sdir.store %c, %C[$i]
}