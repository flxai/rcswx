use serde_json::Value;

pub fn assert_frozen_contract(actual: &Value, expected: &Value) {
    let supported = |mut capture: Value| {
        if let Some(nontrivial) = capture["analysis"]["nontrivial"].as_array().cloned() {
            // The old exporter called apply(validate=false) with raw prefixes,
            // including zero-cost boundary edits that public selection rejects.
            // Do not freeze the resulting malformed offspring as a contract.
            // All computation traces, plans, samples and supported frames remain
            // compared; forced-wave tests additionally compare complete captures
            // against the current serial engine, including these unsafe frames.
            for frame in capture["analysis"]["frames"].as_array_mut().unwrap() {
                if frame["selected"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|index| !nontrivial.contains(index))
                {
                    frame.as_object_mut().unwrap().remove("result");
                    frame.as_object_mut().unwrap().remove("trace");
                }
            }
        }
        capture
    };
    assert_eq!(supported(actual.clone()), supported(expected.clone()));
}
