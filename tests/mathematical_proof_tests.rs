//! Integration Tests for Mathematical Proof System
//!
//! Constitution: Phase 1, Task 1.1 - Validation Criteria
//!
//! This test suite verifies all mathematical proofs required by the constitution:
//! - Entropy production theorem verified
//! - Information bounds verified
//! - Quantum relations verified
//! - All proofs have analytical + numerical verification

use mathematics::information_theory::{EntropyNonNegativity, MutualInformationNonNegativity};
use mathematics::proof_system::{MathematicalStatement, ProofResult};
use mathematics::quantum_mechanics::HeisenbergUncertainty;
use mathematics::thermodynamics::EntropyProductionTheorem;

#[test]
fn test_entropy_production_theorem() {
    let theorem = EntropyProductionTheorem::new();
    let result = theorem.verify();

    match result {
        ProofResult::Verified {
            analytical,
            numerical,
            confidence,
        } => {
            assert!(
                analytical,
                "Entropy production: analytical verification failed"
            );
            assert!(
                numerical,
                "Entropy production: numerical verification failed"
            );
            assert!(
                confidence > 0.999,
                "Entropy production: low confidence {}",
                confidence
            );
        }
        _ => panic!(
            "Entropy production theorem verification failed: {}",
            result
        ),
    }

    // Verify LaTeX representation
    assert_eq!(theorem.latex(), r"\frac{dS}{dt} \geq 0");

    // Verify assumptions
    let assumptions = theorem.assumptions();
    assert_eq!(
        assumptions.len(),
        3,
        "Expected 3 assumptions for entropy production"
    );
    assert!(
        assumptions.iter().all(|a| a.verified),
        "All assumptions should be verified"
    );
}

#[test]
fn test_information_entropy_non_negativity() {
    let theorem = EntropyNonNegativity::new();
    let result = theorem.verify();

    match result {
        ProofResult::Verified {
            analytical,
            numerical,
            confidence,
        } => {
            assert!(
                analytical,
                "Information entropy: analytical verification failed"
            );
            assert!(
                numerical,
                "Information entropy: numerical verification failed"
            );
            assert!(
                confidence > 0.999,
                "Information entropy: low confidence {}",
                confidence
            );
        }
        _ => panic!("Information entropy verification failed: {}", result),
    }

    // Verify LaTeX representation
    assert!(theorem.latex().contains("H(X)"));
    assert!(theorem.latex().contains(r"\geq 0"));
}

#[test]
fn test_mutual_information_non_negativity() {
    let theorem = MutualInformationNonNegativity::new();
    let result = theorem.verify();

    match result {
        ProofResult::Verified {
            analytical,
            numerical,
            confidence,
        } => {
            assert!(
                analytical,
                "Mutual information: analytical verification failed"
            );
            assert!(
                numerical,
                "Mutual information: numerical verification failed"
            );
            assert!(
                confidence > 0.999,
                "Mutual information: low confidence {}",
                confidence
            );
        }
        _ => panic!("Mutual information verification failed: {}", result),
    }

    // Verify LaTeX representation
    assert!(theorem.latex().contains("I(X;Y)"));
}

#[test]
fn test_heisenberg_uncertainty() {
    let theorem = HeisenbergUncertainty::new();
    let result = theorem.verify();

    match result {
        ProofResult::Verified {
            analytical,
            numerical,
            confidence,
        } => {
            assert!(
                analytical,
                "Heisenberg uncertainty: analytical verification failed"
            );
            assert!(
                numerical,
                "Heisenberg uncertainty: numerical verification failed"
            );
            assert!(
                confidence > 0.999,
                "Heisenberg uncertainty: low confidence {}",
                confidence
            );
        }
        _ => panic!("Heisenberg uncertainty verification failed: {}", result),
    }

    // Verify LaTeX representation
    assert!(theorem.latex().contains(r"\Delta x"));
    assert!(theorem.latex().contains(r"\Delta p"));
    assert!(theorem.latex().contains(r"\hbar"));

    // Verify assumptions
    let assumptions = theorem.assumptions();
    assert_eq!(
        assumptions.len(),
        4,
        "Expected 4 assumptions for Heisenberg uncertainty"
    );
    assert!(
        assumptions.iter().all(|a| a.verified),
        "All assumptions should be verified"
    );
}

#[test]
fn test_all_theorems_pass() {
    // Constitution requirement: All fundamental theorems must verify
    let result = mathematics::verify_all_theorems();
    assert!(
        result.is_ok(),
        "All theorems must pass verification: {:?}",
        result
    );
}

#[test]
fn test_theorem_descriptions() {
    let thermo = EntropyProductionTheorem::new();
    assert!(!thermo.description().is_empty());

    let info = EntropyNonNegativity::new();
    assert!(!info.description().is_empty());

    let quantum = HeisenbergUncertainty::new();
    assert!(!quantum.description().is_empty());
}

#[test]
fn test_theorem_domains() {
    let thermo = EntropyProductionTheorem::new();
    assert!(!thermo.domain().is_empty());

    let info = EntropyNonNegativity::new();
    assert!(!info.domain().is_empty());

    let quantum = HeisenbergUncertainty::new();
    assert!(!quantum.domain().is_empty());
}

/// Constitution Validation: All proofs must have both analytical and numerical verification
#[test]
fn test_dual_verification_requirement() {
    let theorems: Vec<Box<dyn MathematicalStatement>> = vec![
        Box::new(EntropyProductionTheorem::new()),
        Box::new(EntropyNonNegativity::new()),
        Box::new(MutualInformationNonNegativity::new()),
        Box::new(HeisenbergUncertainty::new()),
    ];

    for theorem in theorems {
        match theorem.verify() {
            ProofResult::Verified {
                analytical,
                numerical,
                ..
            } => {
                assert!(
                    analytical,
                    "Theorem must have analytical verification: {}",
                    theorem.description()
                );
                assert!(
                    numerical,
                    "Theorem must have numerical verification: {}",
                    theorem.description()
                );
            }
            result => panic!(
                "Theorem verification failed for {}: {}",
                theorem.description(),
                result
            ),
        }
    }
}

/// Performance test: Verification should complete in reasonable time
#[test]
fn test_verification_performance() {
    use std::time::Instant;

    let start = Instant::now();
    let _ = mathematics::verify_all_theorems();
    let duration = start.elapsed();

    // All proofs should complete in under 30 seconds
    assert!(
        duration.as_secs() < 30,
        "Verification took too long: {:?}",
        duration
    );
}

/// Test that proofs are deterministic (same result on repeated runs)
#[test]
fn test_proof_determinism() {
    let theorem = EntropyProductionTheorem::new();

    let result1 = theorem.verify();
    let result2 = theorem.verify();

    assert_eq!(
        result1, result2,
        "Proof verification should be deterministic"
    );
}
