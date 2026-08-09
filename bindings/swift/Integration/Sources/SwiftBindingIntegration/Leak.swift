import Libdictenstein
import Liblevenshtein

// C9 leak-discipline suite for the Swift facade. Swift uses ARC (deterministic
// reference counting, no tracing GC), so a leaked wrapper is one that is not
// deallocated when its last strong reference drops. Each cycle takes a `weak`
// reference to the facade objects and, after their scope ends, asserts the weak
// reference is nil -- the ARC/deinit-equivalent signal that the object was
// reclaimed. Over >=10,000 cycles a retained handle (or a retain cycle) would
// surface as a non-nil weak reference. Native handles are freed at close().

enum LeakTests {
    static func run() throws {
        let fixture: [(String, UInt64?)] = [("cat", 1), ("cot", 2), ("cut", 3), ("scat", nil)]

        for _ in 0..<10_000 {
            let dictionary = try DynamicDAWG()
            for (term, value) in fixture { _ = try dictionary.put(term, value: value) }
            weak var weakTransducer: Transducer?
            weak var weakCursor: QueryCursor?
            do {
                let transducer = try Transducer(dictionary: dictionary)
                weakTransducer = transducer
                let cursor = try transducer.query("cat", maximumDistance: 2)
                weakCursor = cursor
                for _ in cursor {}
                cursor.close()
                transducer.close()
            }
            precondition(weakTransducer == nil, "Transducer was not deallocated (leak)")
            precondition(weakCursor == nil, "QueryCursor was not deallocated (leak)")
            dictionary.close()
        }

        for _ in 0..<10_000 {
            weak var weakPattern: PhoneticPattern?
            weak var weakRules: PhoneticRuleSet?
            do {
                let pattern = try PhoneticPattern.regex("c[ao]t")
                weakPattern = pattern
                _ = try pattern.matches("cat")
                pattern.close()
                let rules = try PhoneticRuleSet.builtin(.englishOrthography)
                weakRules = rules
                _ = try rules.apply("phone")
                rules.close()
            }
            precondition(weakPattern == nil, "PhoneticPattern was not deallocated (leak)")
            precondition(weakRules == nil, "PhoneticRuleSet was not deallocated (leak)")
        }

        print("Swift leak tests passed")
    }
}
