#!/bin/bash
# Archive legacy documents that conflict with constitution

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Legacy Document Archival${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Create legacy archive directory
echo -e "${YELLOW}Creating legacy_docs/ archive...${NC}"
mkdir -p legacy_docs

# Documents to archive (contain conflicts or outdated info)
LEGACY_DOCS=(
    "COMPLETE_SYSTEM_STATUS.md"
    "STATUS.md"
    "CODE_AUDIT_REPORT.md"
    "REPORT_VALIDITY_ASSESSMENT.md"
    "FINAL_HONEST_ASSESSMENT.md"
)

ARCHIVED_COUNT=0

for doc in "${LEGACY_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "${YELLOW}Archiving:${NC} $doc"
        mv "$doc" legacy_docs/
        ARCHIVED_COUNT=$((ARCHIVED_COUNT + 1))
    else
        echo -e "${YELLOW}Skipping:${NC} $doc (not found)"
    fi
done

# Create archive README
cat > legacy_docs/README.md << 'EOF'
# Legacy Documentation Archive

⚠️ **WARNING**: These documents are from pre-constitution development.

They are preserved for **historical reference only** and may contain:
- Outdated architecture descriptions
- Incorrect performance claims
- Forbidden pseudoscience terminology
- Superseded design decisions

## ❌ DO NOT USE THESE AS IMPLEMENTATION GUIDES

The **ONLY** authoritative source is:
**`../IMPLEMENTATION_CONSTITUTION.md v1.0.0`**

## Why These Are Archived

These documents were created before the constitution was established.
They may conflict with current requirements in:
- Scientific rigor (thermodynamic violations)
- Terminology (forbidden pseudoscience terms)
- Architecture (superseded by ADRs)
- Performance claims (unverified)

## Purpose of This Archive

These documents are kept for:
- Historical reference
- Understanding prior work
- Tracking project evolution
- Learning from past approaches

They do **NOT** represent:
- Current project direction
- Valid implementation guides
- Approved architectures
- Accurate performance data

## Current Authoritative Documents

See project root:
- IMPLEMENTATION_CONSTITUTION.md (supreme authority)
- PROJECT_STATUS.md (current progress)
- PROJECT_DASHBOARD.md (live metrics)
- ARCHITECTURE_DECISIONS.md (approved designs)
- TESTING_STRATEGY.md (quality standards)

---

**Last Updated**: 2024-01-28
**Archived By**: Constitution compliance cleanup
EOF

# Fix remaining documents with minor violations
echo -e "${YELLOW}Fixing minor terminology issues...${NC}"

if [ -f "docs/ALGORITHM_COMPARISON_LKH.md" ]; then
    sed -i 's/Traditional Thinking:/Traditional Approach:/g' docs/ALGORITHM_COMPARISON_LKH.md
    sed -i 's/Traditional HPC Thinking:/Traditional HPC Approach:/g' docs/ALGORITHM_COMPARISON_LKH.md
    echo -e "${GREEN}✓${NC} Fixed docs/ALGORITHM_COMPARISON_LKH.md"
fi

echo ""
echo -e "${GREEN}✓${NC} Archived $ARCHIVED_COUNT legacy documents"
echo -e "${GREEN}✓${NC} Created legacy_docs/README.md with disclaimer"
echo -e "${GREEN}✓${NC} Fixed terminology in remaining documents"
echo ""

# Verify cleanup
echo -e "${YELLOW}Verifying cleanup...${NC}"
REMAINING_VIOLATIONS=$(grep -r -i "sentient\|conscious\|self-aware" --include="*.md" . 2>/dev/null | \
    grep -v "IMPLEMENTATION_CONSTITUTION.md" | \
    grep -v "GOVERNANCE_SETUP_COMPLETE.md" | \
    grep -v "legacy_docs/" | \
    grep -v "DARPA_Narcissus/SOLICITATION" | \
    wc -l)

if [ "$REMAINING_VIOLATIONS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No forbidden terms remain in active documents"
else
    echo -e "${YELLOW}⚠${NC}  Found $REMAINING_VIOLATIONS remaining violations"
    echo "Run: grep -r -i 'sentient\\|conscious' --include='*.md' . | grep -v legacy_docs/"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Legacy document cleanup complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Archived documents moved to: legacy_docs/"
echo "Constitution compliance: Ready for Phase 1"
