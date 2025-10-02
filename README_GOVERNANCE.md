# Governance System - Quick Start Guide

## 🎯 What This Is

A **persistent implementation strategy and governance system** for the Active Inference Platform that ensures:
- Constitution-compliant development across all sessions
- Scientific rigor and mathematical correctness
- Protection against scope creep and pseudoscience
- Automated tracking and validation
- Full context for AI-assisted development

---

## 🚀 Quick Start (Every Session)

### For AI Assistants (Claude, GPT, etc.)

**Run this command:**
```bash
./scripts/load_context.sh
```

**Copy the output and paste it to your AI assistant.** It will include:
- Constitution verification ✅
- Current phase and task
- Project status
- Mandatory rules
- Development guidelines

### For Human Developers

**Start every work session:**
```bash
# Load context
./scripts/load_context.sh

# View current task
cat .ai-context/current-task.md

# View progress
cat PROJECT_DASHBOARD.md
```

---

## 📁 Key Files

### **IMPLEMENTATION_CONSTITUTION.md** 🔒
- **THE** supreme authority
- 12-week development plan
- Cannot be modified (git hook protection)
- Version: 1.0.0
- Hash: ca7d9a8d1671a2d46bbcbdf72186d43c353aabc5be89e954a4d78bb5c536d966

### **PROJECT_STATUS.md**
- Overall project progress
- Phase completion status
- Current blockers
- Next actions

### **PROJECT_DASHBOARD.md**
- Live metrics
- Recent activity
- Validation status
- Performance tracking

### **DISASTER_RECOVERY.md**
- Emergency procedures
- Recovery scenarios
- Escalation protocols
- Contact information

---

## 🛡️ Protection Systems

### Git Hooks (Active)
- **pre-commit**: Blocks constitution modifications and pseudoscience terms
- **post-commit**: Verifies constitution integrity

### Automated Checks
- SHA-256 hash verification
- Forbidden term detection
- Constitution compliance validation

### What's Blocked
- ❌ Direct constitution edits
- ❌ Pseudoscience terms (see forbidden list in constitution)
- ❌ Skipping validation gates
- ❌ CPU fallback code

---

## 📊 Current Status

**Phase 0: Development Environment** - 60% Complete
- ✅ AI context system
- ✅ Constitution created and protected
- ✅ Automation scripts
- ✅ Git hooks active
- ✅ Tracking dashboards
- ✅ Recovery procedures
- ⏳ Validation framework (next)
- ⏳ Compliance engine (pending)
- ⏳ Testing strategy (pending)
- ⏳ Quality enforcement (pending)

**Next Milestone**: Complete Phase 0, begin Phase 1 (Mathematical Foundations)

---

## 🎓 Development Workflow

### 1. Start Session
```bash
./scripts/load_context.sh
```

### 2. Check Current Task
```bash
cat .ai-context/current-task.md
```

### 3. Implement Following Constitution
```bash
# Find your task in the constitution
cat IMPLEMENTATION_CONSTITUTION.md | grep -A 50 "Task [X.Y]"
```

### 4. Validate Before Committing
```bash
cargo test --all
cargo clippy -- -D warnings
# Git hooks will automatically check the rest
```

### 5. Commit with Constitution Reference
```bash
git commit -m "feat(phaseX.taskY): Description

Constitution: Phase X Task Y.Z
Validation: PASSED
..."
```

### 6. Update Progress
```bash
vim PROJECT_STATUS.md
vim .ai-context/current-task.md
```

---

## 🔧 Common Commands

### View Constitution
```bash
cat IMPLEMENTATION_CONSTITUTION.md | less
```

### Check Compliance
```bash
sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256
```

### View Progress
```bash
cat PROJECT_DASHBOARD.md
```

### Emergency Recovery
```bash
cat DISASTER_RECOVERY.md
```

### Run All Validations
```bash
cargo test --all
cargo bench
cargo clippy -- -D warnings
```

---

## ✅ Validation Gates

All code must pass these before proceeding:

1. **Mathematical Correctness**
   - Functions have proofs
   - Edge cases handled
   - Numerical stability verified

2. **Scientific Accuracy**
   - Thermodynamic laws respected
   - Information bounds satisfied
   - Quantum constraints met

3. **Performance Contracts**
   - Latency requirements met
   - Throughput targets achieved
   - GPU utilization adequate

4. **Code Quality**
   - Tests passing (>95% coverage)
   - No clippy warnings
   - Documentation complete

---

## 🚨 Emergency Procedures

### Constitution Corruption
```bash
cat DISASTER_RECOVERY.md | grep -A 30 "Constitution File Corruption"
```

### System Not Working
```bash
cat DISASTER_RECOVERY.md | grep -A 30 "Common Issues"
```

### Need to Rollback
```bash
git checkout [LAST_VALIDATED_TAG]
```

---

## 📝 Amendment Process

To update the constitution (rare):

1. Create new version:
   ```bash
   cp IMPLEMENTATION_CONSTITUTION.md IMPLEMENTATION_CONSTITUTION_v1.1.0.md
   ```

2. Edit new version file (not original)

3. Propose changes with:
   - Full justification
   - Impact analysis
   - Team review

4. After unanimous approval:
   ```bash
   mv IMPLEMENTATION_CONSTITUTION_v1.1.0.md IMPLEMENTATION_CONSTITUTION.md
   sha256sum IMPLEMENTATION_CONSTITUTION.md > IMPLEMENTATION_CONSTITUTION.md.sha256
   git add -A
   git commit --no-verify -m "docs: Update constitution to v1.1.0"
   ```

---

## 💡 Tips for Success

### For AI Assistants
- Always start by loading context
- Reference constitution sections
- Validate against requirements
- Don't deviate from plan

### For Human Developers
- Trust the process
- Follow the phases in order
- Don't skip validation gates
- Update tracking documents

### For Teams
- Constitution is supreme authority
- When in doubt, consult it
- Keep metrics updated
- Regular compliance checks

---

## 📞 Support

### Questions About Governance?
Read: `IMPLEMENTATION_CONSTITUTION.md`

### Questions About Current Task?
Read: `.ai-context/current-task.md`

### System Not Working?
Read: `DISASTER_RECOVERY.md`

### Need Help?
1. Check dashboard: `cat PROJECT_DASHBOARD.md`
2. Check status: `cat PROJECT_STATUS.md`
3. Load context: `./scripts/load_context.sh`

---

## 🎉 Success!

The governance infrastructure is complete and operational.

**You can now:**
- Start any session with full context
- Have AI assistants follow the exact plan
- Track progress systematically
- Enforce scientific rigor automatically
- Recover from any disaster
- Maintain constitution compliance

**Next session:**
```bash
./scripts/load_context.sh
```

Then continue with Phase 0 completion and move into Phase 1!

---

**System Status**: 🟢 OPERATIONAL
**Constitution**: 🔒 PROTECTED
**Ready for**: Production Development
