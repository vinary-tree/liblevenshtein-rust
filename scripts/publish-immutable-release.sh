#!/usr/bin/env bash
set -euo pipefail

die() { echo "immutable release: $*" >&2; exit 1; }

verify_assets() {
  local release=$1 tag=$2 assets=$3 state=$4
  local count name file digest api_digest
  [[ -d "$assets" ]] || die "asset directory missing"
  count=$(find "$assets" -mindepth 1 -maxdepth 1 -type f | wc -l)
  [[ "$count" -gt 0 ]] || die "empty asset directory"
  if [[ "$state" == draft ]]; then
    jq -e --arg tag "$tag" --argjson count "$count" \
      '.tag_name == $tag and .draft == true and (.assets | length) == $count' \
      <<< "$release" >/dev/null || die "draft release identity or asset count differs"
  else
    jq -e --arg tag "$tag" --argjson count "$count" \
      '.tag_name == $tag and .draft == false and .immutable == true and (.assets | length) == $count' \
      <<< "$release" >/dev/null || die "published release is not immutable or asset count differs"
  fi
  for file in "$assets"/*; do
    [[ -f "$file" ]] || die "non-file asset in staging directory"
    name=$(basename "$file")
    [[ $(jq -r --arg name "$name" \
      '[.assets[] | select(.name == $name and .state == "uploaded")] | length' \
      <<< "$release") == 1 ]] || die "asset absent, duplicated, or not uploaded: $name"
    digest=$(sha256sum "$file")
    digest=${digest%% *}
    api_digest=$(jq -r --arg name "$name" \
      '.assets[] | select(.name == $name) | .digest' <<< "$release")
    [[ "$api_digest" == "sha256:$digest" ]] ||
      die "release API digest differs from staged bytes: $name"
  done
}

check_tag() {
  local repository=$1 tag=$2 source_sha=$3 refs remote_sha
  [[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ &&
     "$tag" =~ ^v[0-9A-Za-z._-]+$ &&
     "$source_sha" =~ ^[0-9a-f]{40}$ ]] || die "invalid source identity"
  [[ $(git rev-parse HEAD) == "$source_sha" ]] ||
    die "checkout differs from workflow source commit"
  refs=$(git ls-remote --tags "https://github.com/$repository.git" \
    "refs/tags/$tag" "refs/tags/$tag^{}")
  remote_sha=$(awk -v ref="refs/tags/$tag^{}" '$2 == ref { print $1 }' <<< "$refs")
  if [[ -z "$remote_sha" ]]; then
    remote_sha=$(awk -v ref="refs/tags/$tag" '$2 == ref { print $1 }' <<< "$refs")
  fi
  [[ "$remote_sha" == "$source_sha" ]] ||
    die "public tag does not resolve to workflow source commit"
}

case "${1:-}" in
  --verify-local)
    [[ $# == 5 ]] || die "usage: $0 --verify-local <release-json> <tag> <asset-dir> <draft|published>"
    [[ "$5" == draft || "$5" == published ]] || die "invalid release state"
    verify_assets "$(cat "$2")" "$3" "$4" "$5"
    ;;
  --preflight)
    [[ $# == 4 ]] || die "usage: $0 --preflight <repo> <tag> <source-sha>"
    [[ -n "${GH_TOKEN:-}" && -n "${IMMUTABLE_RELEASES_READ_TOKEN:-}" ]] ||
      die "publisher token or Administration:read preflight token missing"
    check_tag "$2" "$3" "$4"
    enabled=$(GH_TOKEN="$IMMUTABLE_RELEASES_READ_TOKEN" \
      gh api "repos/$2/immutable-releases" --jq '.enabled') ||
      die "cannot verify repository immutable-release setting"
    [[ "$enabled" == true ]] || die "immutable releases are not enabled"
    response=$(gh api -i "repos/$2/releases/tags/$3" 2>&1 || true)
    status=$(awk '/^HTTP/ { code=$2 } END { print code }' <<< "$response")
    [[ "$status" == 404 ]] ||
      die "release already exists or cannot prove its absence (HTTP $status)"
    ;;
  --publish)
    [[ $# == 6 ]] || die "usage: $0 --publish <repo> <tag> <asset-dir> <source-sha> <new-readback-dir>"
    [[ -n "${GH_TOKEN:-}" ]] || die "publisher token missing"
    repository=$2 tag=$3 assets=$4 source_sha=$5 readback=$6
    [[ ! -e "$readback" ]] || die "readback directory already exists"
    check_tag "$repository" "$tag" "$source_sha"
    release=$(gh api "repos/$repository/releases/tags/$tag")
    verify_assets "$release" "$tag" "$assets" draft
    gh release edit "$tag" --repo "$repository" --draft=false --verify-tag
    release=$(gh api "repos/$repository/releases/tags/$tag")
    verify_assets "$release" "$tag" "$assets" published
    mkdir -p "$readback"
    for file in "$assets"/*; do
      name=$(basename "$file")
      gh release download "$tag" --repo "$repository" \
        --pattern "$name" --dir "$readback"
      cmp -s "$file" "$readback/$name" ||
        die "published asset bytes differ from staged bytes: $name"
    done
    echo "published immutable release $repository $tag with verified public assets"
    ;;
  *)
    die "usage: $0 --preflight|--publish|--verify-local ..."
    ;;
esac
