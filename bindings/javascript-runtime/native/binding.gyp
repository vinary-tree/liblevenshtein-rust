{
  "variables": {
    "vinary_tree_profile%": "debug"
  },
  "targets": [{
    "target_name": "vinary_tree_native",
    "sources": ["src/addon.cc"],
    "include_dirs": [
      "../../../include",
      "../../../vinary-tree-interop/include",
      "../../../../libdictenstein/include",
      "../../../../lling-llang/include",
      "../../../../duallity/include"
    ],
    "cflags_cc": ["-std=c++20", "-Wall", "-Wextra", "-Werror"],
    "conditions": [["OS=='linux'", {
      "libraries": [
        "-Wl,--start-group",
        "<(module_root_dir)/../../../../duallity/target/<(vinary_tree_profile)/libduallity.a",
        "<(module_root_dir)/../../../../lling-llang/target/<(vinary_tree_profile)/liblling_llang.a",
        "<(module_root_dir)/../../../target/<(vinary_tree_profile)/libliblevenshtein.a",
        "<(module_root_dir)/../../../../libdictenstein/target/<(vinary_tree_profile)/liblibdictenstein.a",
        "-Wl,--end-group",
        "-ldl",
        "-lpthread",
        "-lm"
      ]
    }], ["OS=='mac'", {
      "libraries": [
        "<(module_root_dir)/../../../../duallity/target/<(vinary_tree_profile)/libduallity.a",
        "<(module_root_dir)/../../../../lling-llang/target/<(vinary_tree_profile)/liblling_llang.a",
        "<(module_root_dir)/../../../target/<(vinary_tree_profile)/libliblevenshtein.a",
        "<(module_root_dir)/../../../../libdictenstein/target/<(vinary_tree_profile)/liblibdictenstein.a",
        "-liconv",
        "-framework CoreFoundation",
        "-framework Security"
      ]
    }], ["OS=='win'", {
      "libraries": [
        "<(module_root_dir)/../../../../duallity/target/<(vinary_tree_profile)/duallity.lib",
        "<(module_root_dir)/../../../../lling-llang/target/<(vinary_tree_profile)/lling_llang.lib",
        "<(module_root_dir)/../../../target/<(vinary_tree_profile)/liblevenshtein.lib",
        "<(module_root_dir)/../../../../libdictenstein/target/<(vinary_tree_profile)/libdictenstein.lib",
        "bcrypt.lib",
        "userenv.lib",
        "ws2_32.lib",
        "ntdll.lib",
        "synchronization.lib",
        "advapi32.lib"
      ]
    }]]
  }]
}
