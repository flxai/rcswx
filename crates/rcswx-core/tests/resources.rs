use rcswx_core::architecture::{Budget, Limits};
use rcswx_core::error::Error;
use rcswx_core::recursive::{self, Failure, Token};

fn tokens(leaf: &str) -> Vec<Token> {
    vec![
        Token {
            id: 0,
            name: "start_node".into(),
            children: vec![],
            parent_arity: 0,
        },
        Token {
            id: 1,
            name: "computation".into(),
            children: vec![leaf.into()],
            parent_arity: 0,
        },
    ]
}

#[test]
fn allocation_budget_interrupts_before_kernel_host_work() {
    let first = tokens("identity");
    let second = tokens("relu");
    let mut host_calls = 0;
    let mut host = || {
        host_calls += 1;
        Ok(())
    };
    let mut budget = Budget {
        limits: Limits {
            max_allocation_bytes: Some(0),
            ..Limits::default()
        },
        work: 0,
        output: 0,
        allocation_bytes: 0,
        check: &mut host,
        wait_check: None,
    };
    let mut cause = None;
    let result = recursive::align_observed(&first, &second, false, &mut |stats| match budget
        .account(0, 0, stats.allocation_bytes)
    {
        Ok(()) => Ok(()),
        Err(error) => {
            cause = Some(error);
            Err(Failure::Callback)
        }
    });
    assert!(matches!(result, Err(Failure::Callback)));
    assert_eq!(cause, Some(Error::Limit("allocation")));
    assert_eq!(host_calls, 0);
}

#[test]
fn output_budget_rejects_export_instead_of_truncating_histories() {
    let first = tokens("identity");
    let second = tokens("relu");
    let complete = recursive::align(&first, &second, false, &mut || Ok(())).unwrap();
    let mut requested = 0;
    let result = recursive::align_observed(&first, &second, false, &mut |stats| {
        requested = stats.output_steps;
        if stats.output_steps > 0 {
            Err(Failure::Callback)
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(Failure::Callback)));
    assert_eq!(requested, complete.paths[0].len());
    assert_eq!(complete.distance, 0.5);
}

#[test]
fn host_cancellation_is_not_a_reference_or_numeric_failure() {
    let mut host = || Err(Error::Cancelled);
    let mut budget = Budget {
        limits: Limits::default(),
        work: 0,
        output: 0,
        allocation_bytes: 0,
        check: &mut host,
        wait_check: None,
    };
    assert_eq!(budget.checkpoint(1, 0, 0), Err(Error::Cancelled));
    assert_eq!(budget.work, 1);
}
