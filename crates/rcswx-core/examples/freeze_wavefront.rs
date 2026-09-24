//! Explicit, one-time export. Never invoke this from fixture/build/test targets.
#[path = "../tests/support/wavefront_oracle.rs"]
mod oracle;

use serde_json::{Value, json};

fn main() {
    let mut args = std::env::args().skip(1);
    let output = args
        .next()
        .expect("usage: freeze_wavefront NEW_OUTPUT_PATH SOURCE_REVISION");
    let revision = args
        .next()
        .expect("an explicit source revision is required");
    assert!(args.next().is_none(), "unexpected extra arguments");
    assert!(
        !std::path::Path::new(&output).exists(),
        "refusing to overwrite a frozen oracle"
    );
    let source: Value = serde_json::from_str(include_str!(
        "../../../examples/web/fixtures/recordings.json"
    ))
    .unwrap();
    let mut inputs = source["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            (
                pair["id"].as_str().unwrap().to_owned(),
                pair["parent1_json"].as_str().unwrap().to_owned(),
                pair["parent2_json"].as_str().unwrap().to_owned(),
            )
        })
        .collect::<Vec<_>>();
    let tree = |name: &str, child: Option<&str>| {
        let nodes = match child {
            Some(child) => {
                json!([{"id":"0","name":name,"children":[1]},{"id":"1","name":child,"children":[]}])
            }
            None => json!([{"id":"0","name":name,"children":[]}]),
        };
        json!({"schema":1,"grammar":"oracle","grammar_version":"1","root":0,"nodes":nodes})
            .to_string()
    };
    inputs.push((
        "ordered-ties".into(),
        tree("sequential", Some("relu")),
        tree("sequential", Some("sigmoid")),
    ));
    inputs.push((
        "empty-parameter-error".into(),
        tree("relu", None),
        tree("relu", None),
    ));
    let mut repeated: Value = serde_json::from_str(&inputs[0].1).unwrap();
    for node in repeated["nodes"].as_array_mut().unwrap() {
        node["id"] = json!("0");
    }
    inputs.push((
        "repeated-source-identities".into(),
        repeated.to_string(),
        inputs[0].2.clone(),
    ));
    let mut cases = Vec::new();
    for (name, first, second) in inputs {
        for collapse in [false, true] {
            for max_events in [50_000, 19] {
                cases.push(json!({
                    "name":name,"first":first,"second":second,
                    "collapse_corners":collapse,"max_events":max_events,
                    "expected":oracle::capture(&first,&second,collapse,max_events),
                }));
            }
        }
    }
    let frozen = json!({"schema":1,"source_revision":revision,"generator":"cargo run -p rcswx-core --features trace --example freeze_wavefront -- NEW_OUTPUT_PATH SOURCE_REVISION","cases":cases});
    let bytes = serde_json::to_vec(&frozen).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    use std::io::Write;
    file.write_all(&bytes).unwrap();
    println!(
        "froze {} cases in {} bytes",
        frozen["cases"].as_array().unwrap().len(),
        bytes.len()
    );
}
