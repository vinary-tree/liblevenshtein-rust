use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{
    BincodeSerializer, DatProtobufSerializer, DictionarySerializer, GzipSerializer,
    OptimizedProtobufSerializer, ProtobufSerializer, SuffixAutomatonProtobufSerializer,
};
use libdictenstein::suffix_automaton::SuffixAutomaton;
use std::fs::{self, File};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).expect("output directory argument");
    fs::create_dir_all(&root)?;
    let dictionary = DoubleArrayTrie::from_terms(["", "cab", "café"]);
    BincodeSerializer::serialize(&dictionary, File::create(format!("{root}/dictionary-bincode-v1.bin"))?)?;
    ProtobufSerializer::serialize(&dictionary, File::create(format!("{root}/dictionary-protobuf-v1.bin"))?)?;
    OptimizedProtobufSerializer::serialize(&dictionary, File::create(format!("{root}/dictionary-protobuf-v2.bin"))?)?;
    DatProtobufSerializer::serialize_dat(&dictionary, File::create(format!("{root}/dictionary-dat-v1.bin"))?)?;
    GzipSerializer::<BincodeSerializer>::serialize(&dictionary, File::create(format!("{root}/dictionary-gzip-bincode-v1.bin"))?)?;
    GzipSerializer::<ProtobufSerializer>::serialize(&dictionary, File::create(format!("{root}/dictionary-gzip-protobuf-v1.bin"))?)?;
    GzipSerializer::<OptimizedProtobufSerializer>::serialize(&dictionary, File::create(format!("{root}/dictionary-gzip-protobuf-v2.bin"))?)?;
    let suffix = SuffixAutomaton::from_texts(["banana", "bandana"]);
    BincodeSerializer::serialize_suffix_automaton(&suffix, File::create(format!("{root}/suffix-bincode-v1.bin"))?)?;
    SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(&suffix, File::create(format!("{root}/suffix-protobuf-v1.bin"))?)?;
    Ok(())
}
