//! Reference-compatible structural token preparation.
//!
//! The token stream deliberately retains source logical identity separately
//! from occurrence indices. In particular, equal decimal IDs are legal and
//! boundary tokens use the ID of the wrapper occurrence that owns them.

use crate::architecture::Architecture;
use crate::error::{Error, Result};
use crate::recursive::Token;
use num_bigint::BigInt;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct PreparedToken {
    pub token: Token,
    /// The architecture occurrence represented by this token. The synthetic
    /// start token has no occurrence; boundary tokens point at their wrapper.
    pub occurrence: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct PreparedPair {
    pub first: Architecture,
    pub second: Architecture,
    pub first_tokens: Vec<PreparedToken>,
    pub second_tokens: Vec<PreparedToken>,
    /// Dense kernel IDs mapped back to their lossless decimal source IDs.
    pub identities: Vec<String>,
    /// New second-parent IDs indexed by second.nodes occurrence.
    pub parent2_ids: Vec<String>,
    /// The first unused decimal ID after second-parent renumbering.
    pub next_id: String,
}

#[derive(Clone, Copy)]
enum Event {
    Visit(usize),
    Boundary { owner: usize, branch: usize },
}

fn decimal_id(value: &str) -> Result<BigInt> {
    BigInt::from_str(value)
        .map_err(|_| Error::InvalidInput("logical IDs must be decimal integers".into()))
}

fn identity(
    value: &str,
    lookup: &mut HashMap<String, i64>,
    identities: &mut Vec<String>,
) -> Result<i64> {
    if let Some(&value) = lookup.get(value) {
        return Ok(value);
    }
    let dense = i64::try_from(identities.len()).map_err(|_| Error::Limit("logical identity"))?;
    lookup.try_reserve(1).map_err(|_| Error::Memory)?;
    identities.try_reserve(1).map_err(|_| Error::Memory)?;
    let source = value.to_owned();
    lookup.insert(source.clone(), dense);
    identities.push(source);
    Ok(dense)
}

fn token_for(
    architecture: &Architecture,
    occurrence: usize,
    parents: &[Option<usize>],
    lookup: &mut HashMap<String, i64>,
    identities: &mut Vec<String>,
) -> Result<PreparedToken> {
    let node = architecture.nodes.get(occurrence).ok_or(Error::Index)?;
    let mut children = Vec::new();
    children
        .try_reserve(node.children.len())
        .map_err(|_| Error::Memory)?;
    for &child in &node.children {
        children.push(
            architecture
                .nodes
                .get(child)
                .ok_or(Error::Index)?
                .name
                .clone(),
        );
    }
    let parent_arity = parents
        .get(occurrence)
        .ok_or(Error::Index)?
        .map(|parent| architecture.nodes[parent].children.len())
        .unwrap_or(0);
    Ok(PreparedToken {
        token: Token {
            id: identity(&node.id, lookup, identities)?,
            name: node.name.clone(),
            children,
            parent_arity,
        },
        occurrence: Some(occurrence),
    })
}

fn boundary_for(
    architecture: &Architecture,
    owner: usize,
    branch: usize,
    lookup: &mut HashMap<String, i64>,
    identities: &mut Vec<String>,
) -> Result<PreparedToken> {
    let node = architecture.nodes.get(owner).ok_or(Error::Index)?;
    let final_branch = branch.checked_add(1).ok_or(Error::Index)?
        == node.children.len().checked_sub(1).ok_or(Error::Index)?;
    Ok(PreparedToken {
        token: Token {
            id: identity(&node.id, lookup, identities)?,
            name: if final_branch {
                "wrap_end".into()
            } else {
                "wrap_sep".into()
            },
            children: Vec::new(),
            parent_arity: node.children.len(),
        },
        occurrence: Some(owner),
    })
}

fn breakdown(
    architecture: &Architecture,
    lookup: &mut HashMap<String, i64>,
    identities: &mut Vec<String>,
) -> Result<Vec<PreparedToken>> {
    let parents = architecture.parents();
    let extra = architecture
        .nodes
        .len()
        .checked_mul(2)
        .ok_or(Error::Memory)?;
    let mut result = Vec::new();
    result.try_reserve(extra).map_err(|_| Error::Memory)?;
    let mut events = Vec::new();
    events.try_reserve(extra).map_err(|_| Error::Memory)?;
    events.push(Event::Visit(architecture.root));

    while let Some(event) = events.pop() {
        match event {
            Event::Boundary { owner, branch } => {
                result.push(boundary_for(
                    architecture,
                    owner,
                    branch,
                    lookup,
                    identities,
                )?);
            }
            Event::Visit(occurrence) => {
                let node = architecture.nodes.get(occurrence).ok_or(Error::Index)?;
                let included = parents[occurrence]
                    .map(|parent| !architecture.nodes[parent].name.contains("computation"))
                    .unwrap_or(true);
                if !included {
                    continue;
                }
                if !node.name.contains("sequential") {
                    result.push(token_for(
                        architecture,
                        occurrence,
                        &parents,
                        lookup,
                        identities,
                    )?);
                }
                if node.children.len() > 2 {
                    // A LIFO event stack emits each subtree immediately before
                    // its closing boundary, exactly like Python list rebinding.
                    for branch in (1..node.children.len() - 1).rev() {
                        events.push(Event::Boundary {
                            owner: occurrence,
                            branch,
                        });
                        events.push(Event::Visit(node.children[branch]));
                    }
                } else {
                    for &child in node.children.iter().rev() {
                        events.push(Event::Visit(child));
                    }
                }
            }
        }
    }
    Ok(result)
}

fn start_token(id: i64) -> PreparedToken {
    PreparedToken {
        token: Token {
            id,
            name: "start_node".into(),
            children: Vec::new(),
            parent_arity: 0,
        },
        occurrence: None,
    }
}

/// Tokenize one validated architecture without assigning or mutating parent
/// IDs. The returned dense identity table maps each token ID back to its
/// original external decimal logical ID.
pub fn tokenize_architecture(
    architecture: &Architecture,
) -> Result<(Vec<PreparedToken>, Vec<String>)> {
    architecture.validate()?;
    let mut lookup = HashMap::new();
    let mut identities = Vec::new();
    // The synthetic sentinel has a dense identity, but is not in the source-ID
    // lookup. A real occurrence with external ID "-1" must remain distinct.
    identities.try_reserve(1).map_err(|_| Error::Memory)?;
    identities.push("-1".into());
    let capacity = architecture
        .nodes
        .len()
        .checked_mul(2)
        .and_then(|size| size.checked_add(1))
        .ok_or(Error::Memory)?;
    let mut tokens = Vec::new();
    tokens.try_reserve(capacity).map_err(|_| Error::Memory)?;
    tokens.push(start_token(0));
    tokens.extend(breakdown(architecture, &mut lookup, &mut identities)?);
    Ok((tokens, identities))
}

/// Snapshot both parents, assign fresh arbitrary-width IDs to parent two, and
/// produce the exact stream consumed by the recursive kernel. It never mutates
/// a caller-owned architecture; compatibility adapters apply `parent2_ids` at
/// their documented mutation boundary.
pub fn prepare(first: Architecture, second: Architecture) -> Result<PreparedPair> {
    prepare_internal(first, second, None)
}

/// Prepare host-owned trees with physical occurrence identity supplied outside
/// their opaque metadata. Equal origins model the legacy parent's ID effects.
pub fn prepare_with_origins(
    first: Architecture,
    second: Architecture,
    first_origins: &[usize],
    second_origins: &[usize],
) -> Result<PreparedPair> {
    if first_origins.len() != first.nodes.len() || second_origins.len() != second.nodes.len() {
        return Err(Error::InvalidInput(
            "physical origins must match architecture occurrences".into(),
        ));
    }
    prepare_internal(first, second, Some((first_origins, second_origins)))
}

fn prepare_internal(
    mut first: Architecture,
    mut second: Architecture,
    origins: Option<(&[usize], &[usize])>,
) -> Result<PreparedPair> {
    first.validate()?;
    second.validate()?;

    let mut ids = first.nodes.iter();
    let first_id = ids.next().ok_or(Error::Index)?;
    let mut next = decimal_id(&first_id.id)?;
    for node in ids {
        let candidate = decimal_id(&node.id)?;
        if candidate > next {
            next = candidate;
        }
    }
    next += 1;

    let order = second.preorder();
    let mut parent2_ids = Vec::new();
    parent2_ids
        .try_reserve(second.nodes.len())
        .map_err(|_| Error::Memory)?;
    for _ in 0..second.nodes.len() {
        parent2_ids.push(String::new());
    }
    let mut assigned_origins = HashMap::new();
    if origins.is_some() {
        assigned_origins
            .try_reserve(second.nodes.len())
            .map_err(|_| Error::Memory)?;
    }
    for occurrence in order {
        let assigned = next.to_string();
        let node = second.nodes.get_mut(occurrence).ok_or(Error::Index)?;
        node.id = assigned.clone();
        *parent2_ids.get_mut(occurrence).ok_or(Error::Index)? = assigned.clone();
        if let Some((_, second_origins)) = origins {
            assigned_origins.insert(second_origins[occurrence], occurrence);
        }
        next += 1;
    }
    if let Some((first_origins, _)) = origins {
        for (node, origin) in first.nodes.iter_mut().zip(first_origins) {
            if let Some(&occurrence) = assigned_origins.get(origin) {
                node.id = parent2_ids[occurrence].clone();
            }
        }
    }

    let mut lookup = HashMap::new();
    let mut identities = Vec::new();
    identities.try_reserve(1).map_err(|_| Error::Memory)?;
    identities.push("-1".into());
    let mut first_tokens = Vec::new();
    first_tokens.try_reserve(1).map_err(|_| Error::Memory)?;
    first_tokens.push(start_token(0));
    first_tokens.extend(breakdown(&first, &mut lookup, &mut identities)?);

    let mut second_tokens = Vec::new();
    second_tokens.try_reserve(1).map_err(|_| Error::Memory)?;
    second_tokens.push(start_token(0));
    second_tokens.extend(breakdown(&second, &mut lookup, &mut identities)?);

    Ok(PreparedPair {
        first,
        second,
        first_tokens,
        second_tokens,
        identities,
        parent2_ids,
        next_id: next.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{Node, SCHEMA_VERSION};
    use serde_json::{json, value::to_raw_value};

    fn architecture(ids: &[&str], names: &[&str], children: &[&[usize]]) -> Architecture {
        Architecture {
            schema: SCHEMA_VERSION,
            grammar: "einspace".into(),
            grammar_version: "1".into(),
            root: 0,
            nodes: ids
                .iter()
                .zip(names)
                .zip(children)
                .map(|((id, name), children)| Node {
                    id: (*id).into(),
                    name: (*name).into(),
                    children: children.to_vec(),
                    parameters: None,
                    provenance: None,
                })
                .collect(),
            input_spec: None,
        }
    }

    #[test]
    fn prepares_boundaries_repeated_ids_and_unbounded_updates() {
        let first = architecture(
            &[
                "123456789012345678901234567890",
                "7",
                "7",
                "8",
                "9",
                "10",
                "11",
            ],
            &[
                "branching(2)",
                "clone(2)",
                "computation",
                "computation",
                "add(2)",
                "identity",
                "relu",
            ],
            &[&[1, 2, 3, 4], &[], &[5], &[6], &[], &[], &[]],
        );
        let second = architecture(&["-9", "-8"], &["computation", "identity"], &[&[1], &[]]);

        let prepared = prepare(first, second).unwrap();
        assert_eq!(
            prepared.parent2_ids,
            [
                "123456789012345678901234567891",
                "123456789012345678901234567892"
            ]
        );
        assert_eq!(prepared.second.nodes[0].id, prepared.parent2_ids[0]);
        assert_eq!(prepared.next_id, "123456789012345678901234567893");
        assert_eq!(
            prepared
                .first_tokens
                .iter()
                .map(|token| token.token.name.as_str())
                .collect::<Vec<_>>(),
            [
                "start_node",
                "branching(2)",
                "computation",
                "wrap_sep",
                "computation",
                "wrap_end"
            ]
        );
        assert_eq!(
            prepared
                .first_tokens
                .iter()
                .map(|token| token.occurrence)
                .collect::<Vec<_>>(),
            [None, Some(0), Some(2), Some(0), Some(3), Some(0)]
        );
        assert_eq!(
            prepared.first_tokens[3].token.children,
            Vec::<String>::new()
        );
        assert_eq!(prepared.first_tokens[3].token.parent_arity, 4);
        assert_eq!(
            prepared.first_tokens[1].token.id,
            prepared.first_tokens[3].token.id
        );
    }

    #[test]
    fn applies_shared_legacy_origin_updates_to_first_snapshot() {
        let first = architecture(&["10", "11"], &["computation", "identity"], &[&[1], &[]]);
        let second = architecture(&["4", "5"], &["computation", "relu"], &[&[1], &[]]);

        let prepared = prepare_with_origins(first, second, &[0, 1], &[0, 2]).unwrap();
        assert_eq!(prepared.parent2_ids, ["12", "13"]);
        assert_eq!(prepared.second.nodes[0].id, "12");
        assert_eq!(prepared.first.nodes[0].id, "12");
        assert_eq!(
            prepared.first_tokens[1].token.id,
            prepared.second_tokens[1].token.id
        );
    }

    #[test]
    fn opaque_metadata_cannot_alias_parent_identity() {
        let mut first = architecture(&["10"], &["identity"], &[&[]]);
        let mut second = architecture(&["20"], &["relu"], &[&[]]);
        first.nodes[0].provenance =
            Some(to_raw_value(&json!({"__rcswx_legacy_origin": "shared-root"})).unwrap());
        second.nodes[0].provenance = first.nodes[0].provenance.clone();
        let prepared = prepare(first, second).unwrap();
        assert_eq!(prepared.first.nodes[0].id, "10");
        assert_eq!(prepared.second.nodes[0].id, "11");
        assert_ne!(
            prepared.first_tokens[1].token.id,
            prepared.second_tokens[1].token.id
        );
    }

    #[test]
    fn negative_one_is_a_source_identity_not_the_start_sentinel() {
        let first = architecture(&["-1"], &["identity"], &[&[]]);
        let second = architecture(&["0"], &["relu"], &[&[]]);
        let (tokens, identities) = tokenize_architecture(&first).unwrap();
        assert_ne!(tokens[0].token.id, tokens[1].token.id);
        assert_eq!(identities[tokens[1].token.id as usize], "-1");
        let prepared = prepare(first, second).unwrap();
        assert_ne!(
            prepared.first_tokens[0].token.id,
            prepared.first_tokens[1].token.id
        );
    }

    #[test]
    fn tokenizes_architecture_without_parent_renumbering() {
        let architecture = architecture(
            &["42", "43", "44", "45", "46"],
            &["branching", "clone", "identity", "relu", "add"],
            &[&[1, 2, 3, 4], &[], &[], &[], &[]],
        );
        let original_ids = architecture
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();

        let (tokens, identities) = tokenize_architecture(&architecture).unwrap();

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.token.name.as_str())
                .collect::<Vec<_>>(),
            [
                "start_node",
                "branching",
                "identity",
                "wrap_sep",
                "relu",
                "wrap_end"
            ]
        );
        assert_eq!(identities, ["-1", "42", "44", "45"]);
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.occurrence)
                .collect::<Vec<_>>(),
            [None, Some(0), Some(2), Some(0), Some(3), Some(0)]
        );
        assert_eq!(
            architecture
                .nodes
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            original_ids.iter().map(String::as_str).collect::<Vec<_>>()
        );
    }
}
