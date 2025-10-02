#!/bin/bash
# Automated context loader for AI sessions
# Purpose: Load all necessary context for constitution-compliant development

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Active Inference Platform - Context Loader${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "IMPLEMENTATION_CONSTITUTION.md" ]; then
    echo -e "${RED}ERROR: IMPLEMENTATION_CONSTITUTION.md not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo -e "${GREEN}✓${NC} Found IMPLEMENTATION_CONSTITUTION.md"
echo ""

# Step 1: Verify Constitution Integrity
echo -e "${YELLOW}[1/6]${NC} Verifying Constitution Integrity..."
if sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Constitution integrity verified"
    CONSTITUTION_HASH=$(cat IMPLEMENTATION_CONSTITUTION.md.sha256 | cut -d' ' -f1)
    echo "      SHA-256: $CONSTITUTION_HASH"
else
    echo -e "${RED}✗${NC} Constitution integrity check FAILED!"
    echo "      This is a CRITICAL error. Constitution may have been modified."
    echo "      Follow procedures in DISASTER_RECOVERY.md"
    exit 1
fi
echo ""

# Step 2: Load Project Manifest
echo -e "${YELLOW}[2/6]${NC} Loading Project Manifest..."
if [ -f ".ai-context/project-manifest.yaml" ]; then
    PHASE=$(grep "phase:" .ai-context/project-manifest.yaml | tail -1 | awk '{print $2}')
    TASK=$(grep "task:" .ai-context/project-manifest.yaml | tail -1 | awk '{print $2}' | tr -d '"')
    STATUS=$(grep "status:" .ai-context/project-manifest.yaml | tail -1 | cut -d':' -f2 | xargs)

    echo -e "${GREEN}✓${NC} Project manifest loaded"
    echo "      Current Phase: $PHASE"
    echo "      Current Task: $TASK"
    echo "      Status: $STATUS"
else
    echo -e "${RED}✗${NC} Project manifest not found!"
    exit 1
fi
echo ""

# Step 3: Display Current Task
echo -e "${YELLOW}[3/6]${NC} Current Task Details..."
if [ -f ".ai-context/current-task.md" ]; then
    echo -e "${GREEN}✓${NC} Current task loaded"
    echo ""
    echo "────────────────────────────────────────────────────────────"
    grep "^##" .ai-context/current-task.md | head -5
    echo "────────────────────────────────────────────────────────────"
else
    echo -e "${YELLOW}⚠${NC}  No current task file found"
fi
echo ""

# Step 4: Check for Blockers
echo -e "${YELLOW}[4/6]${NC} Checking for Blockers..."
if [ -f ".ai-context/current-task.md" ]; then
    BLOCKERS=$(grep -A 1 "^## Blockers" .ai-context/current-task.md | tail -1)
    if [ "$BLOCKERS" == "None" ]; then
        echo -e "${GREEN}✓${NC} No blockers"
    else
        echo -e "${RED}⚠${NC}  BLOCKERS DETECTED:"
        echo "      $BLOCKERS"
    fi
else
    echo -e "${GREEN}✓${NC} No blockers file"
fi
echo ""

# Step 5: Check Git Status
echo -e "${YELLOW}[5/6]${NC} Git Repository Status..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    BRANCH=$(git branch --show-current)
    UNCOMMITTED=$(git status --porcelain | wc -l)

    echo -e "${GREEN}✓${NC} Git repository detected"
    echo "      Branch: $BRANCH"

    if [ "$UNCOMMITTED" -gt 0 ]; then
        echo -e "      ${YELLOW}⚠${NC}  Uncommitted changes: $UNCOMMITTED files"
    else
        echo "      Clean working directory"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Not a git repository"
fi
echo ""

# Step 6: Generate AI Context Prompt
echo -e "${YELLOW}[6/6]${NC} Generating AI Context..."
echo -e "${GREEN}✓${NC} Context preparation complete"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   AI ASSISTANT INITIALIZATION PROMPT${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
cat << 'EOF'
I am working on the Active Inference Platform project.

CRITICAL CONTEXT:
- The ONLY implementation guide is IMPLEMENTATION_CONSTITUTION.md
- Constitution Version: 1.0.0
- Constitution Hash: ca7d9a8d1671a2d46bbcbdf72186d43c353aabc5be89e954a4d78bb5c536d966
- All decisions must follow this constitution exactly

CURRENT STATUS:
EOF

echo "- Current Phase: Phase $PHASE"
echo "- Current Task: Task $TASK"
echo "- Status: $STATUS"

cat << 'EOF'

MANDATORY RULES:
1. Constitution is supreme authority
2. No pseudoscience terms (sentient, conscious, aware, etc.)
3. All code must pass validation gates before proceeding
4. Production-grade quality required at all times
5. GPU-first architecture (no CPU fallbacks)
6. Thermodynamic laws must be respected (dS/dt >= 0)
7. Mathematical proofs required for all algorithms
8. Test coverage must exceed 95%

DEVELOPMENT RULES:
- Every function must have mathematical documentation
- Every GPU kernel must document memory access patterns
- Every module must have integration tests
- Error handling must be comprehensive
- Performance contracts must be met

VALIDATION REQUIREMENTS:
Before ANY code is committed:
✓ All tests passing
✓ Validation gates passed
✓ Constitution compliance verified
✓ Performance benchmarks met
✓ Documentation complete

Please confirm you understand and will follow the constitution
before we proceed with any development work.
EOF

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 7: Quick Reference
echo -e "${YELLOW}Quick Reference Commands:${NC}"
echo "  View Constitution:    cat IMPLEMENTATION_CONSTITUTION.md | less"
echo "  View Current Task:    cat .ai-context/current-task.md"
echo "  Run Tests:            cargo test --all"
echo "  Check Compliance:     ./scripts/check_constitution_compliance.sh"
echo "  View Dashboard:       cat PROJECT_DASHBOARD.md"
echo ""

echo -e "${GREEN}Context loading complete. Ready to begin session.${NC}"
echo ""
