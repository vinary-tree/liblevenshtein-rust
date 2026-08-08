(defproject io.vinarytree/liblevenshtein-clojure "0.10.0"
  :description "Idiomatic Clojure facade for snapshot-consistent liblevenshtein FFM bindings"
  :url "https://github.com/vinary-tree/liblevenshtein-rust"
  :license {:name "Apache License 2.0"
            :url "https://www.apache.org/licenses/LICENSE-2.0.txt"}
  :scm {:name "git"
        :url "https://github.com/vinary-tree/liblevenshtein-rust"
        :connection "scm:git:https://github.com/vinary-tree/liblevenshtein-rust.git"
        :developerConnection "scm:git:ssh://git@github.com/vinary-tree/liblevenshtein-rust.git"}
  :pom-addition [:developers
                 [:developer
                  [:id "dylon"]
                  [:name "Dylon Edwards"]
                  [:email "dylon.devo@gmail.com"]]]
  :dependencies [[org.clojure/clojure "1.12.5" :scope "provided"]
                 [io.vinarytree/vinary-tree-interop "0.1.0"]
                 [io.vinarytree/liblevenshtein "0.10.0"]]
  :source-paths ["src"]
  :test-paths ["test"]
  :jvm-opts ["--enable-native-access=ALL-UNNAMED"]
  :filespecs [{:type :path :path "../../LICENSE"}]
  :jar-exclusions [#"(^|/)\.DS_Store$"]
  :deploy-repositories
  [["clojars" {:url "https://repo.clojars.org"
                :username :env/clojars_username
                :password :env/clojars_password
                :sign-releases false}]])
