#!/usr/bin/env bash
set -euo pipefail

die() { echo "immutable release test: $*" >&2; exit 1; }
[[ $# == 1 && ! -e "$1" ]] ||
  die "usage: $0 <new-disk-backed-scratch-dir>"
scratch=$1
mkdir -p "$scratch/assets"
printf 'deterministic release asset\n' > "$scratch/assets/example.tbz"
digest=$(sha256sum "$scratch/assets/example.tbz")
digest=${digest%% *}

jq -n --arg digest "sha256:$digest" \
  '{tag_name:"v4.0.0-rc.6", draft:true, immutable:false,
    assets:[{name:"example.tbz", state:"uploaded", digest:$digest}]}' \
  > "$scratch/draft.json"
./scripts/publish-immutable-release.sh --verify-local \
  "$scratch/draft.json" v4.0.0-rc.6 "$scratch/assets" draft

jq '.draft=false | .immutable=true' "$scratch/draft.json" > "$scratch/published.json"
./scripts/publish-immutable-release.sh --verify-local \
  "$scratch/published.json" v4.0.0-rc.6 "$scratch/assets" published

jq '.assets[0].digest="sha256:0000000000000000000000000000000000000000000000000000000000000000"' \
  "$scratch/draft.json" > "$scratch/bad-digest.json"
if ./scripts/publish-immutable-release.sh --verify-local \
    "$scratch/bad-digest.json" v4.0.0-rc.6 "$scratch/assets" draft; then
  die "accepted a wrong API asset digest"
fi
jq '.assets += [{name:"unexpected.tbz", state:"uploaded", digest:"sha256:none"}]' \
  "$scratch/draft.json" > "$scratch/extra-asset.json"
if ./scripts/publish-immutable-release.sh --verify-local \
    "$scratch/extra-asset.json" v4.0.0-rc.6 "$scratch/assets" draft; then
  die "accepted an unexpected release asset"
fi
if ./scripts/publish-immutable-release.sh --verify-local \
    "$scratch/draft.json" v4.0.0-rc.6 "$scratch/assets" published; then
  die "accepted a mutable or still-draft published release"
fi
echo "immutable release asset identity controls passed"
