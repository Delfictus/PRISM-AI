#!/bin/bash
# Constitution compliance checker
# Validates code against constitution requirements

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Constitution Compliance Checker${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

VIOLATIONS=0

# Check 1: Constitution integrity
echo -e "${YELLOW}[1/5]${NC} Checking constitution integrity..."
if sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Constitution hash matches"
else
    echo -e "${RED}✗${NC} Constitution hash mismatch!"
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check 2: Forbidden terms in source code
echo -e "${YELLOW}[2/5]${NC} Checking for forbidden terms..."
FORBIDDEN_FOUND=0

FORBIDDEN_TERMS=("sentient" "conscious" "self-aware" "thinking" "feeling" "alive" "emergent consciousness" "quantum consciousness")

for term in "${FORBIDDEN_TERMS[@]}"; do
    if grep -r -i "$term" --include="*.rs" src/ 2>/dev/null | grep -v "// Test:" | grep -v "//!" > /dev/null; then
        echo -e "${RED}✗${NC} Found forbidden term: '$term'"
        grep -r -n -i "$term" --include="*.rs" src/ | grep -v "// Test:" | head -5
        FORBIDDEN_FOUND=1
    fi
done

if [ $FORBIDDEN_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No forbidden terms found"
else
    echo -e "${RED}✗${NC} Forbidden terms detected"
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check 3: Required documentation
echo -e "${YELLOW}[3/5]${NC} Checking required documentation..."
REQUIRED_DOCS=(
    "IMPLEMENTATION_CONSTITUTION.md"
    "PROJECT_STATUS.md"
    "PROJECT_DASHBOARD.md"
    "DISASTER_RECOVERY.md"
    "TESTING_STRATEGY.md"
    "ARCHITECTURE_DECISIONS.md"
)

MISSING_DOCS=0
for doc in "${REQUIRED_DOCS[@]}"; do
    if [ ! -f "$doc" ]; then
        echo -e "${RED}✗${NC} Missing required document: $doc"
        MISSING_DOCS=1
    fi
done

if [ $MISSING_DOCS -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All required documents present"
else
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check 4: AI context files
echo -e "${YELLOW}[4/5]${NC} Checking AI context files..."
REQUIRED_CONTEXT=(
    ".ai-context/project-manifest.yaml"
    ".ai-context/development-rules.md"
    ".ai-context/session-init.md"
    ".ai-context/current-task.md"
)

MISSING_CONTEXT=0
for ctx in "${REQUIRED_CONTEXT[@]}"; do
    if [ ! -f "$ctx" ]; then
        echo -e "${RED}✗${NC} Missing context file: $ctx"
        MISSING_CONTEXT=1
    fi
done

if [ $MISSING_CONTEXT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All context files present"
else
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check 5: Git hooks active
echo -e "${YELLOW}[5/5]${NC} Checking git hooks..."
if [ -x ".git/hooks/pre-commit" ] && [ -x ".git/hooks/post-commit" ]; then
    echo -e "${GREEN}✓${NC} Git hooks active"
else
    echo -e "${RED}✗${NC} Git hooks not executable"
    VIOLATIONS=$((VIOLATIONS + 1))
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ $VIOLATIONS -eq 0 ]; then
    echo -e "${GREEN}✅ CONSTITUTION COMPLIANCE: PASSED${NC}"
    echo ""
    echo "All compliance checks passed. System is constitution-compliant."
    exit 0
else
    echo -e "${RED}❌ CONSTITUTION COMPLIANCE: FAILED${NC}"
    echo ""
    echo "Found $VIOLATIONS violation(s). Fix before proceeding."
    echo "See: IMPLEMENTATION_CONSTITUTION.md for requirements"
    exit 1
fi
