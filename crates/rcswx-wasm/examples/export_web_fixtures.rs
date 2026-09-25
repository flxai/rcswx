use rcswx_core::{architecture::Architecture, tokens::prepare, trace::TraceLevel};
use rcswx_wasm::{
    dto::{AnalyzeOptions, Applied},
    session::Session,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fmt::Write, fs, path::PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    pairs: Vec<Pair>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    id: String,
    title: String,
    description: String,
    parent1_json: String,
    parent2_json: String,
}
fn persistent(mut value: Value) -> Value {
    fn strip(v: &mut Value) {
        match v {
            Value::Object(map) => {
                map.remove("plan_id");
                for value in map.values_mut() {
                    strip(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    strip(value)
                }
            }
            _ => {}
        }
    }
    strip(&mut value);
    value
}
fn shape(a: &Architecture, index: usize) -> Value {
    json!([
        a.nodes[index].name,
        a.nodes[index]
            .children
            .iter()
            .map(|&i| shape(a, i))
            .collect::<Vec<_>>()
    ])
}
fn verify_payload(frame: &Applied, first: &Architecture, second: &Architecture) {
    let child = Architecture::from_json(&frame.architecture_json).unwrap();
    assert_eq!(
        child.input_spec.as_ref().map(|v| v.get()),
        second.input_spec.as_ref().map(|v| v.get())
    );
    for origin in &frame.origins {
        if !origin.synthesized {
            assert_eq!(origin.sources.len(), 1);
            let s = &origin.sources[0];
            let source = if s.parent == 1 { first } else { second };
            let node = &source.nodes[s.occurrence_index];
            let output = &child.nodes[origin.occurrence_index];
            assert_eq!(
                node.parameters.as_ref().map(|v| v.get()),
                output.parameters.as_ref().map(|v| v.get())
            );
            assert_eq!(
                node.provenance.as_ref().map(|v| v.get()),
                output.provenance.as_ref().map(|v| v.get())
            );
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = PathBuf::from("examples/web/fixtures/inputs.json");
    let mut output = PathBuf::from("examples/web/fixtures/recordings.json");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--manifest" => manifest = value.into(),
            "--output" => output = value.into(),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let manifest: Manifest = serde_json::from_str(&fs::read_to_string(manifest)?)?;
    assert_eq!(manifest.schema, 1);
    let mut session = Session::new("native-export")?;
    let engine = session.version();
    let options = AnalyzeOptions {
        trace_level: TraceLevel::Full,
        ..Default::default()
    };
    let options_json = serde_json::to_string(&options)?;
    let mut pairs = vec![];
    for pair in manifest.pairs {
        let first = Architecture::from_json(&pair.parent1_json)?;
        let second = Architecture::from_json(&pair.parent2_json)?;
        let prepared = prepare(first.clone(), second.clone())?;
        let analysis = session.analyze(&pair.parent1_json, &pair.parent2_json, &options_json)?;
        let id = &analysis.plan_id;
        let mut frames = vec![];
        for &k in &analysis.prefix_steps {
            let frame = session.preview_step(id, k)?;
            assert_eq!(frame.selected_indices, analysis.nontrivial[..k]);
            verify_payload(&frame, &first, &second);
            frames.push(frame);
        }
        assert_eq!(frames[0].architecture_json, prepared.second.to_json()?);
        let end = Architecture::from_json(&frames.last().unwrap().architecture_json)?;
        assert_eq!(
            shape(&end, end.root),
            shape(&first, first.root),
            "ordered endpoint failed for {}",
            pair.id
        );
        for position in [
            0,
            frames.len() - 1,
            usize::from(frames.len() > 1),
            frames.len() - 1,
            0,
        ] {
            let k = analysis.prefix_steps[position];
            assert_eq!(
                serde_json::to_value(session.preview_step(id, k)?)?,
                serde_json::to_value(&frames[position])?
            );
        }
        let mut samples = vec![];
        for seed in [0u8, 1, 42] {
            let mut bytes = [0u8; 32];
            bytes[0] = seed;
            let mut hex = String::with_capacity(64);
            for byte in bytes {
                write!(hex, "{byte:02x}")?;
            }
            let sample = session.sample(id, &hex, 0.0)?;
            verify_payload(&sample, &first, &second);
            assert_eq!(
                serde_json::to_value(&sample)?,
                serde_json::to_value(session.sample(id, &hex, 0.0)?)?
            );
            samples.push(sample);
        }
        let recording = session.recording(id)?;
        assert!(
            recording.complete,
            "curated fixture recording must be complete"
        );
        println!(
            "{}: {} exact prefixes, {} events, {} trace bytes, endpoint and repeatability passed",
            pair.id,
            frames.len(),
            recording.events.len(),
            recording.bytes
        );
        pairs.push(persistent(json!({"id":pair.id,"title":pair.title,"description":pair.description,"parent1_json":pair.parent1_json,"parent2_json":pair.parent2_json,"input_hashes":{"first":format!("{:x}",Sha256::digest(pair.parent1_json.as_bytes())),"second":format!("{:x}",Sha256::digest(pair.parent2_json.as_bytes()))},"options":options,"analysis":analysis,"recording":recording,"frames":frames,"samples":samples,"endpoint_projection":"ordered operation names and child relationships"})));
        session.dispose(id);
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        output,
        serde_json::to_string(&json!({"schema":1,"engine":engine,"pairs":pairs}))? + "\n",
    )?;
    Ok(())
}
