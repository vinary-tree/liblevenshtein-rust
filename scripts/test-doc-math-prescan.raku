#!/usr/bin/env raku

# Executable contract tests for doc-math-prescan.raku. Static fixtures make the
# accepted and rejected syntax reviewable, while miniature repository roots
# pin append-only exclusion, Rustdoc discovery, and fail-closed path
# classification without creating temporary files.

my $scanner = 'scripts/doc-math-prescan.raku';
my $fixtures = 'tests/fixtures/doc-math';
my $failures = 0;

sub invoke(*@arguments) {
    my $process = run($*EXECUTABLE, $scanner, '--lint', |@arguments, :out, :err);
    my $stdout = $process.out.slurp;
    my $stderr = $process.err.slurp;
    ($process.exitcode, $stdout, $stderr)
}

sub check(Bool $condition, Str $message) {
    unless $condition {
        note "not ok - $message";
        $failures++;
        return;
    }
    say "ok - $message";
}

my ($valid-status, $valid-output, $valid-errors) = invoke(
    "$fixtures/valid.md",
    "$fixtures/valid.rs",
);
check($valid-status == 0, 'valid Markdown and Rustdoc pass');
check($valid-output eq '', 'valid fixtures emit no findings');
check($valid-errors eq '', 'valid fixtures emit no diagnostics');

my ($invalid-status, $invalid-output) = invoke(
    "$fixtures/invalid.md",
    "$fixtures/invalid.rs",
);
check($invalid-status == 1, 'invalid Markdown and Rustdoc fail');
for <bare-dollar-math code-wrapped-dollar-math one-sided-malformed-inline-math
     bare-unicode-math bare-O old-display-dollar-math unicode-in-mathjax
     letter-abuts-open backticked-unicode-math> -> $kind {
    check($invalid-output.contains($kind), "invalid fixtures report $kind");
}

my ($clean-status, $clean-output, $clean-errors) = invoke(
    "--repository-root=$fixtures/repository-clean",
);
check($clean-status == 0, 'repository discovery scans living docs and excludes ledger evidence');
check($clean-output eq '', 'clean fixture repository emits no findings');
check($clean-errors eq '', 'clean fixture repository emits no diagnostics');

my ($unknown-status, $unknown-output) = invoke(
    "--repository-root=$fixtures/repository-unclassified",
);
check($unknown-status == 1, 'unclassified Markdown fails closed');
check(
    $unknown-output.contains('unknown/README.md:0: unclassified-markdown'),
    'unclassified finding names the exact path',
);

if $failures {
    note "$failures doc-math scanner contract test(s) failed";
    exit 1;
}
say 'doc-math scanner contract: all checks passed';
