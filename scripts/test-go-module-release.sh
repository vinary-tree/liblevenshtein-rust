#!/usr/bin/env bash
# Offline disposable-bare-repository tests; never targets the real origin.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
helper="$root/scripts/go-module-release.sh"
mkdir -p "$root/target"
fixture=$(mktemp -d "$root/target/go-module-contract.XXXXXX")
trap 'rm -rf -- "$fixture"' EXIT
repository=vinary-tree/release-fixture
export GITHUB_REPOSITORY=$repository

git init --bare -q "$fixture/origin.git"
git init -q "$fixture/client"
git -C "$fixture/client" config user.name "Go tag contract test"
git -C "$fixture/client" config user.email "go-tag-test@example.invalid"
git -C "$fixture/client" remote add origin "$fixture/origin.git"
mkdir -p "$fixture/client/bindings/go"
printf 'module github.com/%s/bindings/go/v4\n\ngo 1.25\n' "$repository" >"$fixture/client/bindings/go/go.mod"
git -C "$fixture/client" add bindings/go/go.mod
git -C "$fixture/client" commit -qm source
source_sha=$(git -C "$fixture/client" rev-parse HEAD)
git -C "$fixture/client" push -q origin HEAD:refs/heads/main
git -C "$fixture/origin.git" symbolic-ref HEAD refs/heads/main
fixture_source_tag=v4.0.0-rc.6
git -C "$fixture/client" tag -a "$fixture_source_tag" "$source_sha" -m source
git -C "$fixture/client" push -q origin "refs/tags/$fixture_source_tag"

# Mirror the workflow's checkout/ref/peeled-source comparison, including its
# rejection of another dispatch ref or a checkout at a different commit.
source_tag_matches_checkout() {
  local dispatch_ref=$1 checkout_commit
  (
    cd "$fixture/client"
    test "$dispatch_ref" = "$fixture_source_tag" || exit 1
    git fetch --no-tags origin "refs/tags/$fixture_source_tag" >/dev/null || exit 1
    checkout_commit=$(git rev-parse 'HEAD^{commit}') || exit 1
    test "$checkout_commit" = "$(git rev-parse 'FETCH_HEAD^{commit}')"
  )
}
source_tag_matches_checkout "$fixture_source_tag"
if source_tag_matches_checkout another-tag >/dev/null 2>&1; then
  echo "wrong dispatch source ref was accepted" >&2
  exit 1
fi

check() {
  (cd "$fixture/client" && bash "$helper" "$@" "$source_sha")
}
expect_failure() {
  if check "$@" >/dev/null 2>&1; then
    echo "unexpected success: $*" >&2
    exit 1
  fi
}

# Absent is distinguishable from a transport failure and can be created once.
if absent_output=$(check verify bindings/go v4.0.0-rc.6 2>&1); then
  echo "absent ref unexpectedly verified" >&2
  exit 1
else
  status=$?
  [[ $status -eq 2 ]] || {
    printf 'absent ref returned %s, expected 2: %s\n' "$status" "$absent_output" >&2
    exit 1
  }
fi
check create bindings/go v4.0.0-rc.6
before=$(git -C "$fixture/client" ls-remote --refs origin refs/tags/bindings/go/v4.0.0-rc.6)
check verify bindings/go v4.0.0-rc.6
check create bindings/go v4.0.0-rc.6
after=$(git -C "$fixture/client" ls-remote --refs origin refs/tags/bindings/go/v4.0.0-rc.6)
[[ "$before" == "$after" ]] || { echo "repeat create moved tag" >&2; exit 1; }

# A moving ref name or the hash of a tag object is not a literal source commit.
if (cd "$fixture/client" && bash "$helper" verify bindings/go v4.0.0-rc.6 HEAD) >/dev/null 2>&1; then
  echo "moving source ref was accepted" >&2
  exit 1
fi
tag_object=$(git -C "$fixture/client" rev-parse refs/tags/bindings/go/v4.0.0-rc.6)
if (cd "$fixture/client" && bash "$helper" verify bindings/go v4.0.0-rc.6 "$tag_object") >/dev/null 2>&1; then
  echo "tag object ID was accepted as source commit" >&2
  exit 1
fi

# Reject a correctly named annotated tag that points to a different commit.
git -C "$fixture/client" commit --allow-empty -qm other-source
other_sha=$(git -C "$fixture/client" rev-parse HEAD)
if source_tag_matches_checkout "$fixture_source_tag" >/dev/null 2>&1; then
  echo "checkout at wrong source commit was accepted" >&2
  exit 1
fi
git -C "$fixture/client" tag -a bindings/go/v4.0.0-rc.7 "$other_sha" -m wrong-target
git -C "$fixture/client" push -q origin refs/tags/bindings/go/v4.0.0-rc.7
expect_failure verify bindings/go v4.0.0-rc.7
expect_failure create bindings/go v4.0.0-rc.7

# Reject a lightweight ref even when it peels to the correct source.
git -C "$fixture/client" tag bindings/go/v4.0.0-rc.8 "$source_sha"
git -C "$fixture/client" push -q origin refs/tags/bindings/go/v4.0.0-rc.8
expect_failure verify bindings/go v4.0.0-rc.8
expect_failure create bindings/go v4.0.0-rc.8

# Reject an annotated object whose internal tag name differs from its ref.
git -C "$fixture/client" tag -a alternate-name "$source_sha" -m wrong-name
git -C "$fixture/client" push -q origin refs/tags/alternate-name:refs/tags/bindings/go/v4.0.0-rc.9
expect_failure verify bindings/go v4.0.0-rc.9

# Two authorized creators racing for an absent ref must converge, not overwrite.
git -C "$fixture/client" clone -q "$fixture/origin.git" "$fixture/first"
git -C "$fixture/client" clone -q "$fixture/origin.git" "$fixture/second"
for client in first second; do
  git -C "$fixture/$client" config user.name "Go tag contract test"
  git -C "$fixture/$client" config user.email "go-tag-test@example.invalid"
done
(
  cd "$fixture/first"
  bash "$helper" create bindings/go v4.0.0-rc.10 "$source_sha"
) >"$fixture/first.log" 2>&1 &
first_pid=$!
(
  cd "$fixture/second"
  bash "$helper" create bindings/go v4.0.0-rc.10 "$source_sha"
) >"$fixture/second.log" 2>&1 &
second_pid=$!
wait "$first_pid" || { cat "$fixture/first.log" >&2; exit 1; }
wait "$second_pid" || { cat "$fixture/second.log" >&2; exit 1; }
check verify bindings/go v4.0.0-rc.10

# Proxy readback must start from a nonexistent cache, never stale local data.
export GO_MODULE_READBACK_CACHE="$fixture/existing-cache"
mkdir -p "$GO_MODULE_READBACK_CACHE"
expect_failure proxy bindings/go v4.0.0-rc.6

# A stub Go command checks proxy-only environment and exercises JSON readback
# without ever contacting a public registry from this offline test.
mkdir -p "$fixture/mockbin"
export FIXTURE_GOMOD="$fixture/public.go.mod"
printf 'module github.com/%s/bindings/go/v4\n' "$repository" >"$FIXTURE_GOMOD"
# The single quotes intentionally defer expansion to the generated mock.
# shellcheck disable=SC2016
printf '%s\n' \
  '#!/usr/bin/env bash' \
  '[[ "$GOPROXY" == https://proxy.golang.org && -z "$GOPRIVATE" && -z "$GONOPROXY" ]] || exit 1' \
  '[[ "$GOMODCACHE" == */public-cache || "$GOMODCACHE" == */wrong-proxy-cache ]] || exit 1' \
  '[[ "$*" == "mod download -json github.com/$GITHUB_REPOSITORY/bindings/go/v4@v4.0.0-rc.6" ]] || exit 1' \
  'printf "{\"Path\":\"%s\",\"Version\":\"v4.0.0-rc.6\",\"Sum\":\"h1:fixture\",\"GoMod\":\"%s\"}\\n" "${FAKE_GO_MODULE_PATH:-github.com/$GITHUB_REPOSITORY/bindings/go/v4}" "$FIXTURE_GOMOD"' \
  >"$fixture/mockbin/go"
chmod +x "$fixture/mockbin/go"
export PATH="$fixture/mockbin:$PATH"
export GO_MODULE_READBACK_CACHE="$fixture/public-cache"
check proxy bindings/go v4.0.0-rc.6
export FAKE_GO_MODULE_PATH=github.com/example/wrong/v4
export GO_MODULE_READBACK_CACHE="$fixture/wrong-proxy-cache"
expect_failure proxy bindings/go v4.0.0-rc.6
echo "Go module tag state-machine tests passed"
