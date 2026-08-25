require_relative "lib/vinary_tree/liblevenshtein/version"

Gem::Specification.new do |spec|
  spec.name = "liblevenshtein"
  spec.version = VinaryTree::Liblevenshtein::VERSION
  spec.authors = ["Dylon Edwards"]
  spec.email = ["dylon.devo@gmail.com"]
  spec.summary = "Snapshot-consistent streaming Levenshtein search"
  spec.description = "Idiomatic Ruby bindings to Vinary Tree's high-performance liblevenshtein native ABI"
  spec.homepage = "https://github.com/vinary-tree/liblevenshtein-rust"
  spec.license = "Apache-2.0"
  spec.required_ruby_version = ">= 3.3"
  spec.files = Dir["lib/**/*", "README.md", "LICENSE"]
  spec.require_paths = ["lib"]
  spec.metadata = {
    "source_code_uri" => spec.homepage,
    "changelog_uri" => "#{spec.homepage}/blob/main/CHANGELOG.md",
    "rubygems_mfa_required" => "true"
  }
end
