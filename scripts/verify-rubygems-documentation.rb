#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "json"
require "net/http"
require "rubygems/package"
require "uri"

PACKAGE_ONLY = "--package-only"
DEFAULT_ATTEMPTS = 60
DEFAULT_INTERVAL = 10.0
MAX_REDIRECTS = 5

def fail_check(message)
  raise RuntimeError, message
end

def positive_integer_environment(name, default)
  value = Integer(ENV.fetch(name, default.to_s), 10)
  fail_check("#{name} must be positive") unless value.positive?
  value
rescue ArgumentError
  fail_check("#{name} must be an integer")
end

def nonnegative_float_environment(name, default)
  value = Float(ENV.fetch(name, default.to_s))
  fail_check("#{name} must be nonnegative") if value.negative?
  value
rescue ArgumentError
  fail_check("#{name} must be numeric")
end

def get(uri, redirects = MAX_REDIRECTS)
  request = Net::HTTP::Get.new(uri)
  request["User-Agent"] = "vinary-tree-rubygems-documentation-readback/1"
  response = Net::HTTP.start(
    uri.host,
    uri.port,
    use_ssl: uri.scheme == "https",
    open_timeout: 15,
    read_timeout: 30
  ) { |http| http.request(request) }

  if response.is_a?(Net::HTTPRedirection)
    fail_check("too many redirects from #{uri}") unless redirects.positive?
    location = response["location"]
    fail_check("redirect from #{uri} has no location") if location.nil?
    return get(URI.join(uri, location), redirects - 1)
  end

  fail_check("#{uri} returned HTTP #{response.code}") unless response.is_a?(Net::HTTPSuccess)
  response.body
end

def wait_for(label, attempts, interval)
  last_error = "no attempts made"
  attempts.times do |index|
    begin
      yield
      puts "Verified #{label}."
      return
    rescue StandardError => error
      last_error = error.message
      warn "#{label} attempt #{index + 1}/#{attempts}: #{last_error}"
    end
    sleep interval if index + 1 < attempts
  end
  fail_check("#{label} did not converge: #{last_error}")
end

package_only = ARGV.first == PACKAGE_ONLY
ARGV.shift if package_only
gem_path = ARGV.shift or abort "usage: #{$PROGRAM_NAME} [--package-only] GEM [RUBYDOC_PATH=MARKER ...]"
page_checks = ARGV.map do |argument|
  path, marker = argument.split("=", 2)
  abort "RubyDoc checks must use PATH=MARKER" if path.nil? || path.empty? || marker.nil? || marker.empty?
  [path, marker]
end

package = Gem::Package.new(gem_path)
spec = package.spec
version = spec.version.to_s
documentation_uri = "https://www.rubydoc.info/gems/#{spec.name}/#{version}"
fail_check("unexpected documentation_uri") unless spec.metadata.fetch("documentation_uri") == documentation_uri
fail_check("README.md is missing from the gem") unless package.contents.include?("README.md")
fail_check("LICENSE is missing from the gem") unless package.contents.include?("LICENSE")
fail_check("the gem contains no Ruby library sources") unless package.contents.any? { |path| path.match?(%r{\Alib/.+\.rb\z}) }
fail_check("README.md is not an extra RDoc file") unless spec.extra_rdoc_files.include?("README.md")
main_index = spec.rdoc_options.index("--main")
fail_check("RDoc does not select README.md as its main page") unless main_index && spec.rdoc_options[main_index + 1] == "README.md"

puts "Verified packaged documentation for #{spec.name} #{version}."
exit if package_only
abort "at least one PATH=MARKER RubyDoc check is required" if page_checks.empty?

attempts = positive_integer_environment("RUBYGEMS_READBACK_ATTEMPTS", DEFAULT_ATTEMPTS)
interval = nonnegative_float_environment("RUBYGEMS_READBACK_INTERVAL", DEFAULT_INTERVAL)
escaped_name = URI.encode_www_form_component(spec.name)
escaped_version = URI.encode_www_form_component(version)
registry_uri = URI("https://rubygems.org/api/v2/rubygems/#{escaped_name}/versions/#{escaped_version}.json")
expected_sha = Digest::SHA256.file(gem_path).hexdigest

wait_for("RubyGems exact-version metadata", attempts, interval) do
  record = JSON.parse(get(registry_uri))
  fail_check("registry name mismatch") unless record.fetch("name") == spec.name
  fail_check("registry version mismatch") unless record.fetch("version") == version
  fail_check("registry gem checksum mismatch") unless record.fetch("sha") == expected_sha
  fail_check("registry documentation_uri mismatch") unless record.fetch("documentation_uri") == documentation_uri
  metadata = record.fetch("metadata")
  fail_check("registry metadata documentation_uri mismatch") unless metadata.fetch("documentation_uri") == documentation_uri
end

wait_for("RubyDoc exact-version API pages", attempts, interval) do
  index_body = get(URI(documentation_uri))
  fail_check("RubyDoc index omits gem name") unless index_body.include?(spec.name)
  fail_check("RubyDoc index omits gem version") unless index_body.include?(version)
  page_checks.each do |path, marker|
    body = get(URI("#{documentation_uri}/#{path}"))
    fail_check("RubyDoc page #{path} omits #{marker}") unless body.include?(marker)
  end
end
