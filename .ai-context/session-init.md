# Session Initialization Protocol

## START OF SESSION CHECKLIST

### Step 1: Load Context (Required)
```bash
# Run this at the start of every session:
cat .ai-context/project-manifest.yaml
cat IMPLEMENTATION_CONSTITUTION.md | head -100
cat PROJECT_STATUS.md
```

### Step 2: Verify Constitution Integrity
```bash
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256
```
**Expected**: `IMPLEMENTATION_CONSTITUTION.md: OK`

If this fails, STOP immediately and follow DISASTER_RECOVERY.md

### Step 3: Identify Current Position

Check PROJECT_STATUS.md for:
- **Current Phase**: [Will be Phase 0-5]
- **Current Task**: [Will be X.Y format]
- **Last Validated Component**: [Component name]
- **Blocking Issues**: [None or list]

### Step 4: AI Assistant Initialization

**Copy and paste this to AI Assistant:**

```
I am working on the Active Inference Platform project.

CRITICAL CONTEXT:
- The ONLY implementation guide is IMPLEMENTATION_CONSTITUTION.md
- All decisions must follow this constitution exactly
- Current Phase: [INSERT FROM PROJECT_STATUS.md]
- Current Task: [INSERT FROM PROJECT_STATUS.md]

RULES TO FOLLOW:
1. Constitution is supreme authority
2. No pseudoscience terms allowed
3. All code must pass validation gates
4. Production-grade quality required
5. GPU-first architecture mandatory

Please confirm you understand and will follow the constitution before we proceed.
```

### Step 5: Load Current Task Context

```bash
# See what we're working on
cat .ai-context/current-task.md
```

### Step 6: Check Validation Status

```bash
# See recent validation results
cat PROJECT_DASHBOARD.md | grep -A 10 "Validation Failures"
```

---

## CONTEXT HIERARCHY (Most Authoritative First)

1. **IMPLEMENTATION_CONSTITUTION.md** ← SUPREME AUTHORITY
2. **PROJECT_STATUS.md** ← Current state
3. **.ai-context/project-manifest.yaml** ← Project metadata
4. **.ai-context/development-rules.md** ← Coding standards
5. **.ai-context/current-task.md** ← Active work
6. **All other documentation** ← Reference only

### Rule: When documents conflict
Always defer to the document higher in this hierarchy.

---

## OVERRIDE RULES

### The Constitution Overrides Everything
If ANY document, discussion, or decision conflicts with IMPLEMENTATION_CONSTITUTION.md:
- **The constitution wins**
- The conflicting item must be corrected
- Document the conflict in compliance log

### No Exceptions
- Not for "quick fixes"
- Not for "just this once"
- Not for "emergency" situations
- The constitution is immutable

---

## SESSION GOALS

### Before You Start Coding:
- [ ] Constitution section identified
- [ ] Mathematical requirements understood
- [ ] Validation strategy planned
- [ ] Test cases written
- [ ] Performance targets known

### During Development:
- [ ] Follow constitution exactly
- [ ] Write tests first (TDD)
- [ ] Document as you go
- [ ] Validate frequently
- [ ] Check compliance continuously

### Before You Finish:
- [ ] All tests passing
- [ ] Validation gates passed
- [ ] Documentation complete
- [ ] Performance contracts met
- [ ] Constitution compliance verified

---

## COMMON SESSION TYPES

### 1. Implementation Session
**Focus**: Writing new code per constitution

**Checklist**:
- [ ] Loaded constitution section
- [ ] Understood math requirements
- [ ] Tests written first
- [ ] Implementation follows spec
- [ ] Validation gates passed

### 2. Debugging Session
**Focus**: Fixing issues

**Checklist**:
- [ ] Issue doesn't violate constitution
- [ ] Root cause identified
- [ ] Fix maintains compliance
- [ ] Tests updated
- [ ] Regression tests added

### 3. Review Session
**Focus**: Code review

**Checklist**:
- [ ] Constitution compliance verified
- [ ] Mathematical correctness checked
- [ ] Test coverage adequate
- [ ] Performance targets met
- [ ] Documentation complete

### 4. Planning Session
**Focus**: Design and architecture

**Checklist**:
- [ ] Aligns with constitution
- [ ] Phase/task progression valid
- [ ] Dependencies identified
- [ ] Validation strategy defined
- [ ] ADR created if needed

---

## QUICK REFERENCE COMMANDS

### Check Status:
```bash
./scripts/load_context.sh        # Load all context
cat PROJECT_STATUS.md            # Current position
cat PROJECT_DASHBOARD.md         # Live metrics
```

### Validation:
```bash
cargo test --all                 # Run all tests
cargo bench                      # Performance tests
./scripts/check_constitution_compliance.sh  # Compliance check
cargo run --bin validation_gate  # Validation gates
```

### Documentation:
```bash
cargo doc --open                 # View docs
./scripts/generate_docs.sh       # Generate all docs
cat IMPLEMENTATION_CONSTITUTION.md  # Read constitution
```

### Emergency:
```bash
cat DISASTER_RECOVERY.md         # Recovery procedures
git stash                        # Save work
./scripts/verify_constitution_integrity.sh  # Check constitution
```

---

## AI ASSISTANT REMINDERS

### You Should Always:
- Reference specific constitution sections
- Explain mathematical basis
- Include validation steps
- Write comprehensive tests
- Use precise scientific language

### You Should Never:
- Skip validation gates
- Use pseudoscience terms
- Make claims without proofs
- Ignore thermodynamic laws
- Create CPU fallbacks

### When Unsure:
- Stop and ask for clarification
- Reference the constitution
- Don't guess or improvise
- Document the uncertainty
- Wait for explicit guidance

---

## END OF SESSION CHECKLIST

### Before Closing Session:
- [ ] All changes committed
- [ ] Tests passing
- [ ] PROJECT_STATUS.md updated
- [ ] Current task progress documented
- [ ] Blockers noted
- [ ] Constitution compliance verified

### Update These Files:
```bash
# Update status
vim PROJECT_STATUS.md

# Update current task
vim .ai-context/current-task.md

# Commit progress
git add -A
git commit -m "chore(phase.task): Session end update"
```

---

**REMEMBER**: Start EVERY session by loading this file and following the checklist. The constitution is your guide. When in doubt, consult it.
