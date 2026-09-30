#!/usr/bin/env bash
# Immutable subdirectory Go module tag contract. CI uses verify/proxy only;
# an authorized maintainer may run create explicitly after source review.
set -euo pipefail

usage() {
  echo "usage: GITHUB_REPOSITORY=owner/repo $0 {verify|create|proxy} MODULE_DIR VERSION EXPECTED_COMMIT" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
mode=$1
module_dir=$2
version=$3
expected=$4
[[ "$mode" == verify || "$mode" == create || "$mode" == proxy ]] || usage
[[ "$module_dir" =~ ^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_-]+)*$ ]] || usage
[[ "$version" =~ ^v([2-9][0-9]*)\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] || usage
major=${BASH_REMATCH[1]}
[[ "${GITHUB_REPOSITORY:-}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || usage
module="github.com/$GITHUB_REPOSITORY/$module_dir/v$major"
tag="$module_dir/$version"
ref="refs/tags/$tag"
declared=$(sed -n 's/^module //p' "$module_dir/go.mod")
if [[ "$declared" != "$module" ]]; then
  echo "Go module path mismatch: expected $module, found $declared" >&2
  exit 1
fi
expected=$(git rev-parse --verify "$expected^{commit}")
[[ $(git cat-file -t "$expected") == commit ]] || exit 1

# Return 2 only for an absent ref; transport failures are never absence.
remote_ref() {
  local output status sha name
  if output=$(git ls-remote --exit-code --refs origin "$ref"); then
    sha=${output%%$'\t'*}
    name=${output#*$'\t'}
    [[ "$name" == "$ref" && "$sha" =~ ^[0-9a-f]{40,64}$ ]] || {
      echo "unexpected remote ref response: $output" >&2
      return 1
    }
    printf '%s\n' "$sha"
  else
    status=$?
    [[ $status -eq 2 ]] && return 2
    echo "could not inspect remote tag $ref (git exit $status)" >&2
    return 1
  fi
}

check_tag_object() {
  local sha=$1 object kind name peeled
  [[ $(git cat-file -t "$sha") == tag ]] || {
    echo "tag is not an annotated tag object: $ref" >&2
    return 1
  }
  object=$(git cat-file -p "$sha" | sed -n '1s/^object //p')
  kind=$(git cat-file -p "$sha" | sed -n '2s/^type //p')
  name=$(git cat-file -p "$sha" | sed -n '3s/^tag //p')
  peeled=$(git rev-parse "$sha^{}")
  [[ "$object" == "$expected" && "$kind" == commit && "$name" == "$tag" && "$peeled" == "$expected" ]] || {
    echo "annotated tag $ref does not directly name expected commit $expected" >&2
    return 1
  }
}

verify() {
  local remote_sha fetched
  remote_sha=$(remote_ref) || return $?
  git fetch --no-tags origin "$ref" >/dev/null || return 1
  fetched=$(git rev-parse FETCH_HEAD) || return 1
  [[ "$fetched" == "$remote_sha" && $(remote_ref) == "$remote_sha" ]] || {
    echo "remote tag changed during verification: $ref" >&2
    return 1
  }
  check_tag_object "$fetched" || return 1
  echo "verified immutable annotated tag $ref -> $expected"
}

case "$mode" in
  verify)
    verify
    ;;
  create)
    # Do not overwrite either a remote ref or a conflicting local ref.
    if verify; then
      exit 0
    else
      status=$?
      [[ $status -eq 2 ]] || exit "$status"
    fi
    if git show-ref --verify --quiet "$ref"; then
      check_tag_object "$(git rev-parse "$ref")"
    else
      git tag -a "$tag" "$expected" -m "Release $module $version"
    fi
    if git push origin "$ref"; then
      verify
    else
      # A concurrent identical create may have won. Never force or move a ref.
      verify
    fi
    ;;
  proxy)
    verify
    [[ -n "${GO_MODULE_READBACK_CACHE:-}" && ! -e "$GO_MODULE_READBACK_CACHE" ]] || {
      echo "set GO_MODULE_READBACK_CACHE to a fresh, nonexistent directory" >&2
      exit 1
    }
    mkdir -p "$GO_MODULE_READBACK_CACHE"
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
      if result=$(GOWORK=off GOPROXY=https://proxy.golang.org GONOPROXY='' GOPRIVATE='' GONOSUMDB='' \
        GOSUMDB=sum.golang.org GOMODCACHE="$GO_MODULE_READBACK_CACHE" \
        go mod download -json "$module@$version"); then
        path=$(jq -er --arg module "$module" --arg version "$version" \
          'select(.Path == $module and .Version == $version and (.Sum | startswith("h1:"))) | .GoMod' <<<"$result")
        grep -Fx "module $module" "$path"
        echo "public Go proxy resolved $module@$version"
        exit 0
      fi
      [[ $attempt -eq 12 ]] && break
      sleep 10
    done
    echo "public Go proxy did not resolve $module@$version" >&2
    exit 1
    ;;
esac
