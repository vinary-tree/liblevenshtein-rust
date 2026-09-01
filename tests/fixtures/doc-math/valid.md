# Valid mathematical documentation

Inline mathematics uses $`d(x,y) \le k`$, and display mathematics uses a math
fence:

```math
D(x,y)=\min_{a\in A} c(a).
```

The literal dollar token is `$`; regex anchors are written as `^word$`.
Two prices, $5 and $10, are ordinary currency. The term λ-calculus is a name.
Quoted padding strings such as "$$abcabb" and "$$$abcab" are data, not
display delimiters.

```text
Examples inside source fences are literal: $x$, `y ≤ z`, and O(n).
```
