// Binding-free protocol core shared by :vinary and :legacy. Depends on
// NOTHING so it can never leak a dependency across the pair's disjoint
// classpaths.
plugins {
    `java-library`
}
