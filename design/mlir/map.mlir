sdir.map ($i, $j) = 
    (0, 0) to (2, 2) step (1, 1) 
{
    %a = sdir.load %A[$i, $j]
    %b = sdir.load %B[$i, $j]
    %c = sdir.call @add(%a, %b)
    sdir.store %c, %C[$i, $j]
}