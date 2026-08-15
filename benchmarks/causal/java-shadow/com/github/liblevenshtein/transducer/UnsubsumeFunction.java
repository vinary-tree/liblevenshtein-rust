package com.github.liblevenshtein.transducer;

import java.io.Serializable;

/**
 * Measurement-only classpath shadow of the published 3.0.0 implementation.
 *
 * <p>The control flow is a source transcription of the released bytecode. The
 * counters are the only additions. Keeping this class outside production
 * sources lets the probe count actual legacy transitions without modifying or
 * repackaging the published artifact.</p>
 */
public abstract class UnsubsumeFunction implements Serializable {
    private static final long serialVersionUID = 1L;

    protected SubsumesFunction subsumes;

    private static long calls;
    private static long outerPositions;
    private static long comparisons;
    private static long removals;

    public abstract void at(State state, int queryLength);

    public UnsubsumeFunction subsumes(SubsumesFunction function) {
        this.subsumes = function;
        return this;
    }

    public static void resetCounters() {
        calls = 0;
        outerPositions = 0;
        comparisons = 0;
        removals = 0;
    }

    public static long calls() {
        return calls;
    }

    public static long outerPositions() {
        return outerPositions;
    }

    public static long comparisons() {
        return comparisons;
    }

    public static long removals() {
        return removals;
    }

    protected final void measuredAt(State state, int queryLength) {
        calls++;
        StateIterator outer = state.iterator();
        while (outer.hasNext()) {
            Position potentialSubsumer = outer.next();
            outerPositions++;
            int numErrors = potentialSubsumer.numErrors();
            StateIterator inner = outer.copy();

            while (inner.hasNext()) {
                Position candidate = inner.peek();
                if (numErrors < candidate.numErrors()) {
                    break;
                }
                inner.next();
            }

            while (inner.hasNext()) {
                Position candidate = inner.next();
                comparisons++;
                if (subsumes.at(potentialSubsumer, candidate, queryLength)) {
                    inner.remove();
                    removals++;
                }
            }
        }
    }

    public static final class ForStandardPositions extends UnsubsumeFunction {
        private static final long serialVersionUID = 1L;

        @Override
        public void at(State state, int queryLength) {
            measuredAt(state, queryLength);
        }
    }

    public static final class ForSpecialPositions extends UnsubsumeFunction {
        private static final long serialVersionUID = 1L;

        @Override
        public void at(State state, int queryLength) {
            measuredAt(state, queryLength);
        }
    }
}
