use liblevenshtein::transducer::{
    OperationApplicability, OperationSet, OperationType, SubstitutionSet,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).expect("output directory argument");
    fs::create_dir_all(&root)?;
    let mut set = OperationSet::new();
    set.add(OperationType::with_applicability(
        1, 1, 0.0, OperationApplicability::Equal, "match",
    ));
    let mut bytes = SubstitutionSet::new();
    bytes.allow_byte(0xff, 0x80);
    set.add(OperationType::with_restriction(1, 1, 1.0, bytes, "byte"));
    let mut texts = SubstitutionSet::new();
    texts.allow_str("ph", "f");
    set.add(OperationType::with_restriction(2, 1, 0.25, texts, "digraph"));
    fs::write(format!("{root}/operation-binary-v1.bin"), set.to_binary()?)?;
    fs::write(format!("{root}/operation-protobuf-v1.bin"), set.to_protobuf()?)?;
    fs::write(format!("{root}/operation-gzip-binary-v1.bin"), set.to_binary_gzip()?)?;
    fs::write(format!("{root}/operation-gzip-protobuf-v1.bin"), set.to_protobuf_gzip()?)?;
    Ok(())
}
