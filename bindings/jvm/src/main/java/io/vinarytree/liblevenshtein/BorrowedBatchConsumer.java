package io.vinarytree.liblevenshtein;

/** Callback used by the allocation-minimizing borrowed-batch query path. */
@FunctionalInterface
public interface BorrowedBatchConsumer {
    /**
     * Consume one native batch. The batch and all matches become invalid when
     * this method returns and must not escape the callback.
     */
    void accept(BorrowedMatchBatch batch);
}
