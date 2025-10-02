# Disaster Recovery Plan
## Active Inference Platform

**Version**: 1.0.0
**Last Updated**: 2024-01-28
**Review Frequency**: Monthly

---

## Emergency Contact Protocol

### Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | Constitution corrupted, system inoperable | Immediate | All hands |
| **P1 - High** | Validation failures, production broken | < 1 hour | Tech lead |
| **P2 - Medium** | Test failures, performance degradation | < 4 hours | Team |
| **P3 - Low** | Documentation issues, minor bugs | < 24 hours | Individual |

---

## Scenario 1: Constitution File Corruption

### Symptoms
- SHA-256 hash mismatch
- Constitution file modified
- Integrity check fails

### Immediate Actions

```bash
# Step 1: STOP all development immediately
git stash save "Emergency stash - constitution corruption detected"

# Step 2: Verify the issue
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256

# Step 3: Check git history
git log --oneline --follow IMPLEMENTATION_CONSTITUTION.md

# Step 4: Identify last known good version
git log --pretty=format:"%H %s" IMPLEMENTATION_CONSTITUTION.md | head -10

# Step 5: Restore from git
LAST_GOOD_HASH="[INSERT HASH FROM STEP 4]"
git show $LAST_GOOD_HASH:IMPLEMENTATION_CONSTITUTION.md > IMPLEMENTATION_CONSTITUTION.md

# Step 6: Verify restoration
sha256sum IMPLEMENTATION_CONSTITUTION.md
# Expected: ca7d9a8d1671a2d46bbcbdf72186d43c353aabc5be89e954a4d78bb5c536d966

# Step 7: Recommit if hash matches
git add IMPLEMENTATION_CONSTITUTION.md
git commit -m "recovery: Restore constitution from corruption"

# Step 8: Document incident
cat >> INCIDENT_LOG.md << EOF
## Incident: Constitution Corruption
- Date: $(date)
- Detected: SHA-256 mismatch
- Resolution: Restored from commit $LAST_GOOD_HASH
- Root Cause: [TO BE INVESTIGATED]
EOF
```

### Prevention
- ✅ Git pre-commit hook blocks direct modification
- ✅ SHA-256 checksum verification
- ✅ Version control mandatory

---

## Scenario 2: Validation Gate Failures

### Symptoms
- Tests failing
- Validation gates blocking commits
- Performance degradation

### Response Procedure

```bash
# Step 1: Identify failure scope
cargo test --all 2>&1 | tee test_failure.log

# Step 2: Check which validation failed
./scripts/check_constitution_compliance.sh 2>&1 | tee compliance_failure.log

# Step 3: Analyze recent changes
git log --oneline -10
git diff HEAD~1

# Step 4: Rollback to last validated commit
LAST_VALIDATED=$(git tag -l "validated-*" | tail -1)
git checkout $LAST_VALIDATED

# Step 5: Verify rollback
cargo test --all
./scripts/check_constitution_compliance.sh

# Step 6: If tests pass, create recovery branch
git checkout -b recovery/validation-failure-$(date +%Y%m%d)

# Step 7: Cherry-pick good changes
git cherry-pick [GOOD_COMMITS]

# Step 8: Re-run validation
cargo test --all
```

### Prevention
- Mandatory CI/CD validation before merge
- Automated rollback on failure
- Tagged validated checkpoints

---

## Scenario 3: Performance Regression

### Symptoms
- Benchmarks showing >10% slowdown
- Latency exceeding contracts
- GPU utilization dropping

### Response Procedure

```bash
# Step 1: Capture performance baseline
cargo bench --baseline main

# Step 2: Compare current performance
cargo bench

# Step 3: Profile the system
# If GPU available:
nsys profile --trace=cuda,nvtx cargo run --release --example benchmark

# Step 4: Identify bottleneck
# Analyze profile output

# Step 5: Rollback if critical
if [ $PERFORMANCE_REGRESSION -gt 20 ]; then
    git revert HEAD
    cargo bench  # Verify revert fixes performance
fi

# Step 6: Document regression
cat >> PERFORMANCE_LOG.md << EOF
## Performance Regression
- Date: $(date)
- Component: [COMPONENT]
- Regression: ${PERFORMANCE_REGRESSION}%
- Cause: [TO BE DETERMINED]
- Resolution: [REVERTED/OPTIMIZED]
EOF
```

---

## Scenario 4: Thermodynamic Law Violation

### Symptoms
- Entropy production < 0
- Free energy increasing
- Physical law violations detected

### Response Procedure

```bash
# Step 1: This is a CRITICAL scientific error
echo "CRITICAL: Thermodynamic law violation detected"

# Step 2: Immediately halt all operations
# Do NOT commit this code
git stash save "CRITICAL: Thermodynamic violation - do not use"

# Step 3: Identify the violating component
cargo test --package statistical_mechanics -- --nocapture

# Step 4: Review mathematical implementation
# Check oscillator evolution for correct damping term
# Verify thermal noise implementation
# Check entropy calculation

# Step 5: Consult constitution
cat IMPLEMENTATION_CONSTITUTION.md | grep -A 20 "Thermodynamic"

# Step 6: Fix must include mathematical proof
# Create proof that new implementation satisfies dS/dt >= 0

# Step 7: Re-validate with extended testing
cargo test --package thermodynamic_tests -- --test-threads=1

# Step 8: Document in scientific log
cat >> SCIENTIFIC_VALIDATION_LOG.md << EOF
## Thermodynamic Violation Incident
- Date: $(date)
- Component: [COMPONENT]
- Violation: dS/dt = [NEGATIVE_VALUE]
- Mathematical Fix: [PROOF]
- Verification: [TEST_RESULTS]
EOF
```

### Prevention
- Automated thermodynamic validators
- Property-based testing (entropy always increases)
- Mathematical proof required before implementation

---

## Scenario 5: Data Loss or Corruption

### Symptoms
- Working directory corrupted
- Git repository damaged
- Build artifacts lost

### Response Procedure

```bash
# Step 1: Assess damage
git status
git fsck --full

# Step 2: If repository is intact, recover from git
git reflog  # Find lost commits
git checkout [COMMIT_HASH]

# Step 3: If repository damaged, restore from backup
# (Assumes you have regular backups)
# Restore from most recent backup

# Step 4: Verify restoration
./scripts/load_context.sh
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256
cargo build --all

# Step 5: Rebuild artifacts
cargo clean
cargo build --release

# Step 6: Re-run validation
cargo test --all
```

### Prevention
- Regular git backups
- Remote repository synchronization
- Constitution file explicitly backed up

---

## Scenario 6: Phase Progression Error

### Symptoms
- Attempted to start Phase N without completing Phase N-1
- Validation gates bypassed
- Dependencies missing

### Response Procedure

```bash
# Step 1: Check current phase status
cat .ai-context/project-manifest.yaml | grep phase

# Step 2: Verify all prerequisites
cat IMPLEMENTATION_CONSTITUTION.md | grep -A 10 "Phase.*Completion Requires"

# Step 3: Check validation status
cat PROJECT_DASHBOARD.md | grep "Phase.*Complete"

# Step 4: If prerequisites not met, rollback
git checkout phase-$(expr $CURRENT_PHASE - 1)

# Step 5: Complete missing validations
# Follow constitution checklist for previous phase

# Step 6: Update phase status only after validation
vim .ai-context/project-manifest.yaml
# Update phase number

# Step 7: Document corrective action
cat >> COMPLIANCE_LOG.md << EOF
## Phase Progression Correction
- Date: $(date)
- Attempted: Phase $ATTEMPTED
- Current: Phase $CURRENT
- Action: Rolled back to complete prerequisites
EOF
```

---

## Recovery Checklists

### Quick Recovery (< 5 minutes)
- [ ] Constitution integrity verified
- [ ] Git repository functional
- [ ] Can load context successfully
- [ ] Tests passing

### Full Recovery (< 30 minutes)
- [ ] All files restored from git
- [ ] Constitution hash verified
- [ ] All validation gates passing
- [ ] Performance benchmarks met
- [ ] Documentation up to date

### Complete Rebuild (< 2 hours)
- [ ] Fresh clone from repository
- [ ] Constitution verified
- [ ] All dependencies installed
- [ ] Full build successful
- [ ] All tests passing
- [ ] Benchmarks run successfully

---

## Backup Strategy

### What to Backup

**Critical Files (Daily):**
- IMPLEMENTATION_CONSTITUTION.md
- .ai-context/*
- PROJECT_DASHBOARD.md
- PROJECT_STATUS.md
- All source code (src/)

**Important Files (Weekly):**
- All documentation
- Test suites
- Benchmark data
- Validation logs

### Backup Procedure

```bash
# Create timestamped backup
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical files
cp IMPLEMENTATION_CONSTITUTION.md "$BACKUP_DIR/"
cp -r .ai-context "$BACKUP_DIR/"
cp PROJECT_DASHBOARD.md "$BACKUP_DIR/"
cp -r src "$BACKUP_DIR/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"

# Verify archive
tar -tzf "$BACKUP_DIR.tar.gz" | head -20

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Restoration Procedure

```bash
# Extract backup
tar -xzf backups/[TIMESTAMP].tar.gz

# Restore constitution
cp backups/[TIMESTAMP]/IMPLEMENTATION_CONSTITUTION.md .

# Verify
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256

# Restore other files as needed
```

---

## Known Good States

### Constitution Versions

| Version | Date | SHA-256 Hash | Status |
|---------|------|--------------|--------|
| 1.0.0 | 2024-01-28 | ca7d9a8d1671a2d46bbcbdf72186d43c353aabc5be89e954a4d78bb5c536d966 | Current |

### Validated Checkpoints

| Tag | Date | Phase | All Tests | Performance | Notes |
|-----|------|-------|-----------|-------------|-------|
| validated-phase0-complete | TBD | 0 | ✅ | ✅ | Infrastructure ready |
| validated-phase1-complete | TBD | 1 | Pending | Pending | Math foundations |

---

## Escalation Procedures

### Level 1: Self-Recovery (0-30 minutes)
**Who**: Individual developer
**Action**: Follow scenario-specific procedures above
**Escalate if**: Cannot resolve within 30 minutes

### Level 2: Team Recovery (30 minutes - 4 hours)
**Who**: Technical lead + team
**Action**:
1. Review incident log
2. Collaborative debugging
3. Constitution consultation
**Escalate if**: Fundamental architecture issue

### Level 3: Project Reset (4+ hours)
**Who**: Project lead + all stakeholders
**Action**:
1. Project-wide review
2. Constitution amendment consideration
3. Major architectural revision

---

## Post-Incident Protocol

### Incident Report Template

```markdown
# Incident Report: [INCIDENT_TYPE]

## Summary
- **Date**: [DATE]
- **Severity**: [P0/P1/P2/P3]
- **Duration**: [TIME]
- **Impact**: [DESCRIPTION]

## Timeline
- HH:MM - Incident detected
- HH:MM - Recovery initiated
- HH:MM - Issue resolved
- HH:MM - Validation completed

## Root Cause
[Detailed analysis]

## Resolution
[Steps taken]

## Prevention
- [ ] Code changes made
- [ ] Tests added
- [ ] Documentation updated
- [ ] Process improved

## Lessons Learned
[Key takeaways]

## Follow-up Actions
- [ ] Action item 1
- [ ] Action item 2
```

### Incident Log Location
All incidents logged in: `INCIDENT_LOG.md`

---

## Emergency Commands Quick Reference

```bash
# Check constitution integrity
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256

# Load context (get current state)
./scripts/load_context.sh

# Verify system health
cargo test --all
cargo build --all

# Find last validated commit
git tag -l "validated-*" | tail -1

# Rollback to safety
git checkout [LAST_VALIDATED_TAG]

# View incident history
cat INCIDENT_LOG.md

# Check for violations
./scripts/check_constitution_compliance.sh
```

---

## System Health Indicators

### Green (Healthy)
- ✅ Constitution integrity verified
- ✅ All tests passing
- ✅ Performance contracts met
- ✅ No blocking issues

### Yellow (Warning)
- ⚠️ Minor test failures
- ⚠️ Performance within 10% of target
- ⚠️ Documentation incomplete

### Red (Critical)
- 🔴 Constitution corruption
- 🔴 Major test failures
- 🔴 Thermodynamic violations
- 🔴 >20% performance regression

---

## Recovery Testing

After any recovery procedure:

```bash
# Run full validation suite
cargo test --all --verbose
cargo bench
./scripts/check_constitution_compliance.sh
cargo doc --no-deps

# Verify constitution
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256

# Check git integrity
git fsck --full

# Load context (should work)
./scripts/load_context.sh

# If all pass, mark recovery complete
echo "Recovery completed at $(date)" >> RECOVERY_LOG.md
```

---

## Regular Maintenance

### Daily
- [ ] Verify constitution integrity
- [ ] Run test suite
- [ ] Check for uncommitted changes
- [ ] Review dashboard

### Weekly
- [ ] Full validation run
- [ ] Performance benchmarks
- [ ] Backup critical files
- [ ] Review incident log

### Monthly
- [ ] Complete system audit
- [ ] Constitution review (amendments?)
- [ ] Update disaster recovery procedures
- [ ] Archive old backups

---

## Contact Information

### Emergency Contacts
- **Project Lead**: [NAME] - [EMAIL] - [PHONE]
- **Technical Lead**: [NAME] - [EMAIL] - [PHONE]
- **Scientific Advisor**: [NAME] - [EMAIL] - [PHONE]

### Backup Contacts
- **Secondary Technical**: [NAME] - [EMAIL]
- **DevOps**: [NAME] - [EMAIL]

### External Resources
- **DARPA Program Manager**: [IF APPLICABLE]
- **GPU Vendor Support**: NVIDIA Developer Support
- **Cloud Provider**: [IF USING CLOUD]

---

## Recovery Time Objectives (RTO)

| Scenario | Target RTO | Maximum Acceptable |
|----------|------------|-------------------|
| Constitution corruption | 5 minutes | 15 minutes |
| Test failures | 1 hour | 4 hours |
| Performance regression | 4 hours | 1 day |
| Complete data loss | 4 hours | 8 hours |

## Recovery Point Objectives (RPO)

| Data Type | Target RPO | Backup Frequency |
|-----------|------------|------------------|
| Constitution | 0 (git) | Every commit |
| Source code | 0 (git) | Every commit |
| Test data | 1 day | Daily |
| Benchmarks | 1 week | Weekly |

---

## Appendix: Common Issues & Quick Fixes

### Issue: "Cannot find IMPLEMENTATION_CONSTITUTION.md"
**Fix**: `git restore IMPLEMENTATION_CONSTITUTION.md`

### Issue: Context loader fails
**Fix**:
```bash
cd /path/to/project/root
./scripts/load_context.sh
```

### Issue: Git hooks not executing
**Fix**:
```bash
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/post-commit
```

### Issue: Validation gate false positive
**Fix**: Review validation logic, don't bypass gate

### Issue: Performance benchmark timeout
**Fix**: Check GPU availability, reduce benchmark size temporarily

---

**Remember**: When in doubt, STOP and consult this document. Recovery is always possible if you follow these procedures.

**Last Resort**: If all recovery procedures fail, contact project lead immediately with full incident details.
