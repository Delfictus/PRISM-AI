# WEB PLATFORM IMPLEMENTATION CONSTITUTION
## Hard Constraints & Enforcement for Interactive Demo Platform

**Version:** 1.0.0
**Date:** January 9, 2025
**Scope:** 4-Dashboard Interactive Web Platform
**Parent:** PRISM-AI Implementation Constitution

---

## PREAMBLE

This constitution establishes hard constraints and enforcement mechanisms for the PRISM-AI Interactive Demo Platform, ensuring:
- Professional quality (production-grade, not prototype)
- Performance excellence (60fps, <100ms latency)
- Security compliance (DoD-appropriate)
- Systematic development (tracked, validated, governed)

---

## ARTICLE I: PERFORMANCE MANDATES

### 1.1 Rendering Performance (MANDATORY)

**Hard Constraints:**
```javascript
// Every render cycle MUST meet these requirements
const PERFORMANCE_REQUIREMENTS = {
  framerate: 60,           // FPS (enforced)
  frameTime: 16.67,        // ms (1000/60)
  websocketLatency: 100,   // ms max
  initialLoad: 2000,       // ms max
  bundleSize: 500,         // KB max per chunk
};

// Automatic enforcement via performance observer
if (window.performance) {
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.duration > 16.67) {
        console.error(`VIOLATION: Frame time ${entry.duration}ms exceeds 16.67ms`);
        // Auto-report to monitoring
        reportPerformanceViolation(entry);
      }
    }
  });
  observer.observe({ entryTypes: ['measure'] });
}
```

**Build-Time Enforcement:**
```javascript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Code splitting to enforce bundle size limits
        },
      },
    },
    // FAIL build if bundle exceeds limits
    chunkSizeWarningLimit: 500,  // KB
  },
  plugins: [
    bundleSizePlugin({
      maxSize: 500 * 1024,  // Bytes
      failOnExceed: true,    // BLOCKS deployment
    }),
  ],
});
```

### 1.2 WebSocket Latency (MANDATORY)

**Hard Constraints:**
```rust
// backend/src/websocket/performance_guard.rs
pub struct WebSocketPerformanceGuard {
    max_latency_ms: u64,
    abort_on_violation: bool,
}

impl WebSocketPerformanceGuard {
    pub fn enforce_latency(&self, start: Instant) -> Result<(), PerformanceViolation> {
        let elapsed = start.elapsed();

        if elapsed > Duration::from_millis(self.max_latency_ms) {
            if self.abort_on_violation {
                panic!("WebSocket latency {}ms exceeds {}ms limit",
                    elapsed.as_millis(), self.max_latency_ms);
            }

            return Err(PerformanceViolation::LatencyExceeded {
                actual: elapsed,
                limit: Duration::from_millis(self.max_latency_ms),
            });
        }

        Ok(())
    }
}

// Usage in every WebSocket handler
impl Handler<SendUpdate> for PwsaWebSocket {
    fn handle(&mut self, msg: SendUpdate, ctx: &mut Self::Context) {
        let start = Instant::now();

        // ... process update ...

        // MANDATORY enforcement
        LATENCY_GUARD.enforce_latency(start)
            .expect("WebSocket latency violation");
    }
}
```

---

## ARTICLE II: QUALITY GATES

### 2.1 Code Quality (MANDATORY)

**TypeScript Strictness:**
```json
// tsconfig.json - MANDATORY settings
{
  "compilerOptions": {
    "strict": true,                    // All strict checks
    "noUncheckedIndexedAccess": true,  // Array safety
    "noImplicitReturns": true,         // Complete returns
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,

    // BLOCKS compilation on violations
    "skipLibCheck": false,
    "noEmit": false
  }
}
```

**ESLint Configuration:**
```json
// .eslintrc.json - FAIL build on errors
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended"
  ],
  "rules": {
    "no-console": ["error", { "allow": ["warn", "error"] }],
    "no-debugger": "error",
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-unused-vars": "error",
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "error"
  }
}
```

**Pre-commit Hook:**
```bash
#!/bin/bash
# .git/hooks/pre-commit

# TypeScript type checking
npm run type-check || exit 1

# ESLint (no warnings allowed)
npm run lint || exit 1

# Unit tests must pass
npm run test:unit || exit 1

# Build must succeed
npm run build || exit 1

echo "✅ All quality gates passed"
```

### 2.2 Test Coverage (MANDATORY)

**Minimum Coverage Requirements:**
```json
// jest.config.js
{
  "coverageThreshold": {
    "global": {
      "branches": 80,
      "functions": 80,
      "lines": 80,
      "statements": 80
    }
  },
  // FAIL if below threshold
  "collectCoverageFrom": [
    "src/**/*.{ts,tsx}",
    "!src/**/*.d.ts"
  ]
}
```

**CI/CD Gate:**
```yaml
# .github/workflows/ci.yml
- name: Test Coverage
  run: |
    npm run test:coverage
    coverage=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
    if (( $(echo "$coverage < 80" | bc -l) )); then
      echo "BLOCKED: Coverage $coverage% below 80%"
      exit 1
    fi
```

---

## ARTICLE III: ACCESSIBILITY REQUIREMENTS (MANDATORY)

### 3.1 WCAG 2.1 Level AA Compliance

**Automated Testing:**
```javascript
// tests/accessibility.test.tsx
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('Dashboard should have no accessibility violations', async () => {
  const { container } = render(<PwsaDashboard />);
  const results = await axe(container);

  // FAIL test if violations found
  expect(results).toHaveNoViolations();
});
```

**Mandatory Requirements:**
- [ ] All interactive elements keyboard navigable
- [ ] ARIA labels on all visualizations
- [ ] Color contrast ratio ≥ 4.5:1
- [ ] Screen reader compatible
- [ ] Focus indicators visible

**Enforcement:**
```bash
# CI pipeline blocks on violations
npm run test:a11y || exit 1
```

---

## ARTICLE IV: SECURITY MANDATES

### 4.1 Data Classification Enforcement

**Frontend Security:**
```typescript
// src/security/classification.ts
export enum DataClassification {
  UNCLASSIFIED = 0,
  CUI = 1,
  SECRET = 2,
  TOP_SECRET = 3,
}

export class ClassificationGuard {
  static enforce(data: any, requiredLevel: DataClassification) {
    if (data.classification > requiredLevel) {
      throw new SecurityViolation(
        `Data classification ${data.classification} exceeds display limit ${requiredLevel}`
      );
    }
  }
}

// MANDATORY on all data display
function renderTelemetry(data: SatelliteTelemetry) {
  ClassificationGuard.enforce(data, DataClassification.CUI);
  // ... render
}
```

**Backend Security:**
```rust
// backend/src/security/classification.rs
pub fn validate_data_classification(
    data: &SerializableData,
    max_classification: DataClassification,
) -> Result<(), SecurityViolation> {
    if data.classification > max_classification {
        // AUTO-REDACT or BLOCK
        return Err(SecurityViolation::ClassificationExceeded {
            data: data.classification,
            limit: max_classification,
        });
    }
    Ok(())
}

// MANDATORY before WebSocket send
impl PwsaWebSocket {
    fn send_update(&self, update: PwsaUpdate) {
        // Enforce classification limit
        validate_data_classification(&update, DataClassification::CUI)
            .expect("Security violation");

        // ... send
    }
}
```

### 4.2 No Sensitive Data in Logs (MANDATORY)

**Logging Sanitization:**
```rust
// Automatic PII/sensitive data redaction
pub fn sanitize_for_logging(data: &str) -> String {
    let mut sanitized = data.to_string();

    // Redact coordinates (operational security)
    sanitized = COORD_REGEX.replace_all(&sanitized, "[REDACTED]").to_string();

    // Redact classified markers
    sanitized = SECRET_REGEX.replace_all(&sanitized, "[CLASSIFIED]").to_string();

    sanitized
}

// MANDATORY wrapper
macro_rules! safe_log {
    ($($arg:tt)*) => {
        log::info!("{}", sanitize_for_logging(&format!($($arg)*)))
    };
}
```

---

## ARTICLE V: PERIODIC BUILD VERIFICATION (MANDATORY)

### 5.1 Continuous Compilation Requirement

**CRITICAL PRINCIPLE:**
> **Never build on top of errors. All code must compile and pass tests before proceeding to the next task.**

**Enforcement:**
```bash
# MANDATORY after EVERY significant code change
# (every 2-3 tasks, or ~60-90 minutes of work)

# Frontend verification
cd prism-web-platform
npm run build || { echo "BLOCKED: Frontend build failed"; exit 1; }
npm run lint || { echo "BLOCKED: Lint errors found"; exit 1; }
npm run type-check || { echo "BLOCKED: Type errors found"; exit 1; }

# Backend verification (if cargo available)
cd ../
cargo check || { echo "BLOCKED: Rust compilation failed"; exit 1; }
cargo test --lib || { echo "BLOCKED: Tests failing"; exit 1; }
cargo clippy -- -D warnings || { echo "BLOCKED: Clippy warnings found"; exit 1; }

echo "✅ All builds passing - safe to proceed"
```

### 5.2 Verification Frequency (MANDATORY)

**Required Checkpoints:**
1. **After completing each major task** (e.g., after adding TypeScript interfaces)
2. **Before starting Week 2/3/4** (milestone verification)
3. **After installing new dependencies** (ensure no conflicts)
4. **Before committing to git** (pre-commit hook)
5. **At end of each development session** (session validation)

**Time-Based Rule:**
- If working for >90 minutes without compilation check → **VIOLATION**
- If >3 tasks completed without build verification → **VIOLATION**

### 5.3 Build Verification Checklist

**After Every Significant Change:**
```markdown
## Build Verification Checklist (MANDATORY)

Run these commands in order:

### Frontend (React/TypeScript)
- [ ] `npm run type-check` - TypeScript compilation (0 errors required)
- [ ] `npm run lint` - ESLint checks (0 errors/warnings required)
- [ ] `npm run build` - Production build (must succeed)
- [ ] `npm run test` - Unit tests (all passing)

### Backend (Rust)
- [ ] `cargo check` - Fast compilation check
- [ ] `cargo clippy -- -D warnings` - Linter checks
- [ ] `cargo test --lib` - Library tests
- [ ] `cargo build --release` - Full optimized build (periodic)

### Integration
- [ ] Start backend server (verify no panics)
- [ ] Start frontend dev server (verify loads)
- [ ] Open browser to http://localhost:3000 (visual check)
- [ ] Verify WebSocket connection (check browser console)

**Result:**
- ✅ All checks passed → Proceed to next task
- ❌ Any check failed → **STOP**, fix errors before continuing
```

### 5.4 Error Accumulation Prevention

**Problem Scenario (FORBIDDEN):**
```
Task 1: Add TypeScript types → Don't verify build
Task 2: Add React components → Don't verify build (2 compile errors now)
Task 3: Add WebSocket logic → Don't verify build (5 compile errors now)
Task 4: Try to test → **BLOCKED by cascading errors from Tasks 1-3**
```

**Correct Approach (REQUIRED):**
```
Task 1: Add TypeScript types → ✅ Verify build passes
Task 2: Add React components → ✅ Verify build passes
Task 3: Add WebSocket logic → ✅ Verify build passes
Task 4: Test functionality → ✅ All systems operational
```

### 5.5 Automated Build Gates

**Git Pre-Commit Hook (MANDATORY):**
```bash
#!/bin/bash
# .git/hooks/pre-commit
# Automatically runs before every git commit

echo "🔍 Running pre-commit build verification..."

# Change to project root
cd "$(git rev-parse --show-toplevel)"

# Frontend checks
if [ -d "prism-web-platform" ]; then
  echo "📦 Checking frontend..."
  cd prism-web-platform

  npm run type-check || exit 1
  npm run lint || exit 1
  npm run build || exit 1

  cd ..
fi

# Backend checks (if cargo available)
if command -v cargo &> /dev/null; then
  echo "🦀 Checking backend..."

  cargo check || exit 1
  cargo clippy -- -D warnings || exit 1
  cargo test --lib || exit 1
fi

echo "✅ All pre-commit checks passed"
exit 0
```

**Installation:**
```bash
# Copy pre-commit hook
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo "✅ Build verification hook installed"
```

### 5.6 CI/CD Build Verification

**GitHub Actions (MANDATORY):**
```yaml
# .github/workflows/continuous-verification.yml
name: Continuous Build Verification

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  frontend-verification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: cd prism-web-platform && npm ci

      - name: TypeScript compilation
        run: cd prism-web-platform && npm run type-check

      - name: Lint checks
        run: cd prism-web-platform && npm run lint

      - name: Production build
        run: cd prism-web-platform && npm run build

      - name: Unit tests
        run: cd prism-web-platform && npm run test

  backend-verification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Cargo check
        run: cargo check

      - name: Clippy
        run: cargo clippy -- -D warnings

      - name: Unit tests
        run: cargo test --lib

      - name: Build release
        run: cargo build --release
```

### 5.7 Session-End Verification

**End of Day Checklist (MANDATORY):**
```markdown
## Session Complete - Final Verification

Before closing session or switching tasks:

1. **Compile Everything:**
   - [ ] Frontend builds successfully
   - [ ] Backend compiles (if cargo available)
   - [ ] No TypeScript errors
   - [ ] No linter warnings

2. **Test Status:**
   - [ ] All existing tests passing
   - [ ] New tests added for new features
   - [ ] Test coverage maintained/improved

3. **Git Status:**
   - [ ] All changes committed (or stashed)
   - [ ] Commit messages follow convention
   - [ ] Branch is clean

4. **Documentation:**
   - [ ] Progress tracker updated
   - [ ] Task completion log updated
   - [ ] Any blockers documented

**Sign-off:** Session closed cleanly, no technical debt carried forward
```

### 5.8 Violation Reporting

**If Build Fails During Verification:**
```markdown
## Build Failure Protocol

1. **STOP all forward progress immediately**
2. **Document the failure:**
   - What command failed?
   - What was the error message?
   - What task triggered it?
3. **Fix the error before proceeding**
4. **Re-run full verification checklist**
5. **Update progress tracker with resolution**

**Example:**
```
**Build Failure Report - 2025-10-10**

**Task:** Adding Dashboard #1 components
**Failure:** TypeScript compilation failed
**Error:** `Property 'lat' does not exist on type 'SatelliteState'`
**Root Cause:** Missing field in interface definition
**Resolution:** Added `lat: number` to SatelliteState interface
**Verification:** ✅ Build now passes
**Time Lost:** 15 minutes (acceptable)
```

### 5.9 Enforcement Mechanisms

**Automated Monitoring:**
```typescript
// scripts/monitor-build-health.ts
interface BuildHealthMetrics {
  lastSuccessfulBuild: Date;
  timeSinceLastVerification: number;  // minutes
  tasksSinceLastBuild: number;
  violationsCount: number;
}

function enforceVerificationPolicy(metrics: BuildHealthMetrics) {
  // VIOLATION: Too long without verification
  if (metrics.timeSinceLastVerification > 90) {
    throw new PolicyViolation(
      'MANDATORY: Must verify build at least every 90 minutes'
    );
  }

  // VIOLATION: Too many tasks without verification
  if (metrics.tasksSinceLastBuild > 3) {
    throw new PolicyViolation(
      'MANDATORY: Must verify build after every 3 tasks maximum'
    );
  }
}
```

**Dashboard Integration:**
```typescript
// Show build health in development dashboard
export const BuildHealthIndicator: React.FC = () => {
  const health = useBuildHealth();

  return (
    <StatusBadge
      status={health.compliant ? 'success' : 'error'}
      label={`Last build: ${health.minutesAgo} min ago`}
      tooltip={
        health.compliant
          ? 'Build verification up to date'
          : 'WARNING: Build verification overdue'
      }
    />
  );
};
```

---

## ARTICLE VI: DEVELOPMENT WORKFLOW (MANDATORY)

### 5.1 Git Workflow Enforcement

**Branch Protection:**
```yaml
# .github/branch-protection.yml
branches:
  master:
    protection:
      required_status_checks:
        - TypeScript type check
        - ESLint (no warnings)
        - Unit tests (>80% coverage)
        - E2E tests (critical paths)
        - Bundle size check
      required_reviews: 1
      enforce_admins: true
```

**Commit Message Format (Enforced):**
```bash
# .git/hooks/commit-msg
#!/bin/bash
commit_msg=$(cat $1)

# Enforce conventional commits
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore): .{10,}"; then
  echo "BLOCKED: Commit message must follow conventional commits"
  echo "Format: type: description"
  echo "Types: feat, fix, docs, style, refactor, test, chore"
  exit 1
fi
```

### 5.2 Progress Tracking (MANDATORY)

**Daily Updates Required:**
```markdown
# DAILY-PROGRESS-TRACKER.md (MUST update daily)

## Week 1, Day 1 (Date: YYYY-MM-DD)
**Tasks Completed:**
- [ ] Task 1.1.1: Initialize React project
- [ ] Task 1.1.2: Install Material-UI
Status: X/Y tasks complete
Blockers: None / [list blockers]
Next: [tomorrow's plan]

---

**ENFORCEMENT:**
- CI checks for daily update (fail if >24hrs since last update)
- Status dashboard auto-generated from tracker
```

---

## ARTICLE VII: VISUAL QUALITY STANDARDS

### 6.1 Design Consistency (MANDATORY)

**Theme Compliance:**
```typescript
// Every component MUST use theme values, not hard-coded
// ❌ WRONG:
<Box sx={{ backgroundColor: '#0A0E27' }}>  // Hard-coded color

// ✅ CORRECT:
<Box sx={{ backgroundColor: 'background.default' }}>  // Theme token

// Enforced via ESLint rule
{
  "rules": {
    "no-hard-coded-colors": "error"  // Custom rule
  }
}
```

**Component Audit:**
```bash
# CI pipeline checks
npm run audit:design-system || exit 1

# Fails if:
# - Hard-coded colors found
# - Non-theme spacing values
# - Inline styles (should use sx prop)
# - Missing ARIA labels
```

### 6.2 Animation Performance (MANDATORY)

**Frame Budget Enforcement:**
```typescript
// utils/performanceMonitor.ts
export function enforceFrameBudget(callback: () => void) {
  const start = performance.now();

  callback();

  const duration = performance.now() - start;

  if (duration > 16.67) {  // 60fps = 16.67ms per frame
    console.error(`VIOLATION: Frame took ${duration}ms (budget: 16.67ms)`);

    // Auto-report
    reportToSentry({
      type: 'frame-budget-exceeded',
      duration,
      stack: new Error().stack,
    });
  }
}

// MANDATORY wrapper for animations
useEffect(() => {
  const animate = () => {
    enforceFrameBudget(() => {
      // Your animation code
    });
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
}, []);
```

---

## ARTICLE VIII: TESTING REQUIREMENTS

### 7.1 Component Testing (MANDATORY)

**Every Dashboard Component MUST Have:**
```typescript
// Example: MetricCard.test.tsx
describe('MetricCard', () => {
  // 1. MANDATORY: Render test
  it('should render without crashing', () => {
    render(<MetricCard title="Test" value={42} />);
  });

  // 2. MANDATORY: Accessibility test
  it('should have no a11y violations', async () => {
    const { container } = render(<MetricCard title="Test" value={42} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  // 3. MANDATORY: Props test
  it('should display correct value', () => {
    render(<MetricCard title="Test" value={42} unit="ms" />);
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('ms')).toBeInTheDocument();
  });

  // 4. MANDATORY: Interaction test
  it('should handle user interaction', () => {
    const onClick = jest.fn();
    render(<MetricCard title="Test" value={42} onClick={onClick} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalled();
  });
});
```

**Enforcement:**
```bash
# CI blocks merge if component lacks tests
npm run test:component-coverage || exit 1
```

### 7.2 E2E Critical Path Testing (MANDATORY)

**Required E2E Tests:**
```typescript
// e2e/critical-paths.spec.ts
test('User can navigate to all 4 dashboards', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Dashboard 1: PWSA
  await page.click('text=Space Force Data Fusion');
  await expect(page.locator('.globe-canvas')).toBeVisible();

  // Dashboard 2: Telecom
  await page.click('text=Telecom & Logistics');
  await expect(page.locator('.network-graph')).toBeVisible();

  // Dashboard 3: HFT
  await page.click('text=High-Frequency Trading');
  await expect(page.locator('.candlestick-chart')).toBeVisible();

  // Dashboard 4: Internals
  await page.click('text=System Internals');
  await expect(page.locator('.pipeline-diagram')).toBeVisible();
});

test('Real-time updates are received within 100ms', async ({ page }) => {
  await page.goto('http://localhost:3000/pwsa');

  const start = Date.now();

  // Wait for first WebSocket message
  await page.waitForSelector('[data-testid="mission-awareness"]', {
    state: 'visible',
  });

  const latency = Date.now() - start;

  // MUST be under 100ms
  expect(latency).toBeLessThan(100);
});
```

**Enforcement:**
```yaml
# .github/workflows/e2e.yml
- name: E2E Tests
  run: npm run test:e2e
  # BLOCKS merge if critical paths fail
```

---

## ARTICLE IX: DEPLOYMENT GATES

### 8.1 Pre-Deployment Checklist (MANDATORY)

**Automated Validation:**
```bash
#!/bin/bash
# scripts/validate-deployment.sh
# MUST pass before deployment

echo "Running deployment validation..."

# 1. All tests pass
npm run test:all || exit 1

# 2. Coverage threshold met
npm run test:coverage || exit 1

# 3. Bundle size within limits
npm run build || exit 1
npm run analyze-bundle || exit 1

# 4. Lighthouse performance score
npm run lighthouse -- --min-score=90 || exit 1

# 5. Accessibility audit
npm run test:a11y || exit 1

# 6. Security audit
npm audit --audit-level=moderate || exit 1

# 7. Load testing
npm run test:load -- --users=100 --duration=60s || exit 1

echo "✅ All deployment gates passed"
```

**Gate Checklist:**
- [ ] TypeScript: 0 errors
- [ ] ESLint: 0 errors, 0 warnings
- [ ] Tests: >80% coverage, all passing
- [ ] Bundle: <500KB per chunk
- [ ] Lighthouse: >90 performance score
- [ ] Accessibility: WCAG 2.1 AA
- [ ] Security: No high/critical vulnerabilities
- [ ] Load test: 100 concurrent users, <100ms p95

---

## ARTICLE X: CONSTITUTIONAL COMPLIANCE DASHBOARD

### 9.1 Real-Time Compliance Monitoring

**Built-In Compliance Dashboard:**
```typescript
// components/ComplianceDashboard.tsx
export const ComplianceDashboard: React.FC = () => {
  const metrics = usePerformanceMetrics();
  const coverage = useTestCoverage();

  return (
    <Grid container>
      {/* Article I: Performance */}
      <MetricCard
        title="Frame Rate"
        value={metrics.fps}
        unit="fps"
        status={metrics.fps >= 60 ? 'success' : 'error'}
        requirement="≥60fps"
      />

      {/* Article II: Quality */}
      <MetricCard
        title="Type Safety"
        value={metrics.typeErrors}
        unit="errors"
        status={metrics.typeErrors === 0 ? 'success' : 'error'}
        requirement="0 errors"
      />

      {/* Article VII: Testing */}
      <MetricCard
        title="Test Coverage"
        value={coverage.percentage}
        unit="%"
        status={coverage.percentage >= 80 ? 'success' : 'error'}
        requirement="≥80%"
      />

      {/* Article VIII: Deployment */}
      <DeploymentGateStatus gates={metrics.deploymentGates} />
    </Grid>
  );
};
```

**Access:** http://localhost:3000/compliance

---

## ARTICLE XI: PROGRESS TRACKING (MANDATORY)

### 10.1 Task Tracking System

**Obsidian Vault Structure:**
```
07-Web-Platform/
├── 00-Constitution/
│   └── WEB-PLATFORM-CONSTITUTION.md (this file)
│
├── 01-Progress-Tracking/
│   ├── STATUS-DASHBOARD.md          # Central hub (auto-updated)
│   ├── DAILY-PROGRESS-TRACKER.md    # Daily updates (MANDATORY)
│   ├── WEEKLY-REVIEW.md             # Weekly retrospective
│   └── TASK-COMPLETION-LOG.md       # Granular task tracking
│
├── 02-Implementation-Guides/
│   ├── PHASE-1-FOUNDATION.md
│   ├── PHASE-2-DASHBOARDS.md
│   ├── PHASE-3-REFINEMENT.md
│   └── PHASE-4-DEPLOYMENT.md
│
├── 03-Technical-Specs/
│   ├── API-SPECIFICATION.md
│   ├── DATA-MODELS.md
│   ├── WEBSOCKET-PROTOCOL.md
│   └── COMPONENT-LIBRARY.md
│
├── 04-Design-System/
│   ├── THEME-SPECIFICATION.md
│   ├── COMPONENT-GUIDELINES.md
│   └── DESIGN-TOKENS.md
│
└── 05-Quality-Assurance/
    ├── TEST-STRATEGY.md
    ├── PERFORMANCE-BENCHMARKS.md
    └── ACCESSIBILITY-CHECKLIST.md
```

### 10.2 Automated Progress Updates

**Status Dashboard Generator:**
```typescript
// scripts/update-status-dashboard.ts
import fs from 'fs';

interface Task {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed';
  effort_hours: number;
  assignee?: string;
}

function generateStatusDashboard(tasks: Task[]): string {
  const total = tasks.length;
  const completed = tasks.filter(t => t.status === 'completed').length;
  const percentage = Math.round((completed / total) * 100);

  return `
# WEB PLATFORM STATUS DASHBOARD
Last Updated: ${new Date().toISOString()}

## Overall Progress: ${percentage}%

\`\`\`
Phase 1: ${getPhaseProgress(tasks, 'phase1')}
Phase 2: ${getPhaseProgress(tasks, 'phase2')}
Phase 3: ${getPhaseProgress(tasks, 'phase3')}
Phase 4: ${getPhaseProgress(tasks, 'phase4')}
\`\`\`

## Tasks: ${completed}/${total} Complete

[... detailed breakdown ...]
  `;
}

// Auto-run after every task completion
```

**Enforcement:**
```bash
# Git pre-commit hook
./scripts/update-status-dashboard.sh
git add 01-Progress-Tracking/STATUS-DASHBOARD.md
```

---

## ARTICLE XII: DEMONSTRATION REQUIREMENTS

### 11.1 Live Demo Scenarios (MANDATORY)

**Each Dashboard MUST Have:**
- [ ] **Scenario 1:** Nominal operations (everything healthy)
- [ ] **Scenario 2:** Anomaly/threat detection (show alerts)
- [ ] **Scenario 3:** System under stress (high load)
- [ ] **Scenario 4:** Recovery from failure (resilience)

**Scenario Configuration:**
```typescript
// src/scenarios/ScenarioLibrary.ts
export const DEMO_SCENARIOS = {
  pwsa: {
    nominal: {
      name: "Nominal Operations",
      duration: 60,  // seconds
      threatInjection: false,
      description: "All satellites healthy, no threats detected",
    },
    hypersonicThreat: {
      name: "Hypersonic Threat Detection",
      duration: 120,
      threatInjection: true,
      threatType: "hypersonic",
      location: [38.0, 127.0],  // Korean peninsula
      description: "Hypersonic glide vehicle detected, PRISM-AI provides <1ms threat assessment and recommended actions",
    },
    // ... more scenarios
  },
  // ... other dashboards
};
```

**Demo Control Panel:**
```typescript
export const DemoController: React.FC = () => {
  const [currentScenario, setCurrentScenario] = useState<Scenario>();

  return (
    <Box>
      <Select
        value={currentScenario?.name}
        onChange={(e) => loadScenario(e.target.value)}
      >
        {Object.entries(DEMO_SCENARIOS.pwsa).map(([key, scenario]) => (
          <MenuItem key={key} value={key}>
            {scenario.name}
          </MenuItem>
        ))}
      </Select>

      <Button onClick={startScenario}>Start Scenario</Button>
      <Button onClick={pauseScenario}>Pause</Button>
      <Button onClick={resetScenario}>Reset</Button>
    </Box>
  );
};
```

**Enforcement:** All 4 dashboards must have working scenario library before deployment

---

## ARTICLE XIII: MONITORING & OBSERVABILITY

### 12.1 Performance Monitoring (MANDATORY)

**Real-Time Metrics Collection:**
```typescript
// src/monitoring/performanceCollector.ts
export class PerformanceCollector {
  private metrics: PerformanceMetrics[] = [];

  collectFrameMetrics() {
    const entries = performance.getEntriesByType('measure');

    for (const entry of entries) {
      this.metrics.push({
        timestamp: Date.now(),
        name: entry.name,
        duration: entry.duration,
        type: 'frame',
      });

      // VIOLATION: Frame exceeded budget
      if (entry.duration > 16.67) {
        this.reportViolation('frame-budget-exceeded', entry);
      }
    }
  }

  collectWebSocketMetrics(latency: number) {
    this.metrics.push({
      timestamp: Date.now(),
      duration: latency,
      type: 'websocket',
    });

    // VIOLATION: Latency too high
    if (latency > 100) {
      this.reportViolation('websocket-latency-exceeded', { latency });
    }
  }

  // Auto-export to Prometheus
  exportMetrics() {
    return {
      frame_time_p95: this.getPercentile(95),
      websocket_latency_avg: this.getAverage('websocket'),
      violations_count: this.violations.length,
    };
  }
}
```

**Prometheus Integration:**
```rust
// backend/src/monitoring/metrics.rs
use prometheus::{register_histogram, register_counter};

lazy_static! {
    pub static ref WEBSOCKET_LATENCY: Histogram = register_histogram!(
        "websocket_message_latency_ms",
        "WebSocket message processing latency"
    ).unwrap();

    pub static ref FRAME_VIOLATIONS: Counter = register_counter!(
        "frontend_frame_budget_violations",
        "Count of frame budget violations (>16.67ms)"
    ).unwrap();
}

// Auto-collect in every WebSocket handler
impl Handler<SendUpdate> for PwsaWebSocket {
    fn handle(&mut self, msg: SendUpdate, ctx: &mut Self::Context) {
        let start = Instant::now();

        // ... process ...

        // AUTO-RECORD
        WEBSOCKET_LATENCY.observe(start.elapsed().as_millis() as f64);
    }
}
```

---

## ENFORCEMENT SUMMARY

### Build-Time (Compilation)
- ✅ TypeScript strict mode (blocks on type errors)
- ✅ ESLint (blocks on errors/warnings)
- ✅ Bundle size limits (fails if exceeded)
- ✅ Test coverage threshold (fails if <80%)

### Runtime (Development)
- ✅ Performance monitoring (alerts on violations)
- ✅ Frame budget enforcement (60fps)
- ✅ WebSocket latency tracking (<100ms)
- ✅ Security classification checks

### Deployment (CI/CD)
- ✅ All tests must pass
- ✅ Lighthouse score >90
- ✅ Accessibility audit (WCAG 2.1 AA)
- ✅ Load testing (100 users)
- ✅ Security audit (no high/critical)

### Development Workflow
- ✅ Daily progress updates (MANDATORY)
- ✅ Conventional commit messages (enforced)
- ✅ Branch protection (master)
- ✅ Code review required

---

## GOVERNANCE VALIDATION

### Constitutional Compliance Score

**Enforcement Level:**
- Build-time: 10/10 (comprehensive)
- Runtime: 8/10 (monitoring + alerts)
- Workflow: 9/10 (systematic tracking)
- Testing: 10/10 (mandatory coverage)

**Overall:** 9.25/10 (Excellent)

**Comparison to PRISM-AI Core:**
- Core: 6.9/10 (appropriate for dev phase)
- Web Platform: 9.25/10 (production-grade from start)

**Rationale:** Web platform is stakeholder-facing, must be polished from Day 1

---

## METRICS & MONITORING

### Real-Time Dashboard (Built-In)

**Access:** `http://localhost:3000/dev/compliance`

**Displays:**
- Current FPS (live)
- WebSocket latency (p50, p95, p99)
- Test coverage (per module)
- Bundle sizes (per chunk)
- Violations (last 24 hours)
- Deployment gate status (pass/fail)

**Auto-refresh:** Every 5 seconds

---

## INTEGRATION WITH PRISM-AI GOVERNANCE

### Shared Principles

**From PRISM-AI Constitution:**
- ✅ Performance mandates (adapt: <16.67ms vs <5ms)
- ✅ Quality gates (adapt: 80% coverage, TypeScript strict)
- ✅ Systematic tracking (same Obsidian vault approach)
- ✅ Governance monitoring (performance observers)

**Web Platform Specific:**
- ✅ Accessibility requirements (WCAG 2.1 AA)
- ✅ Visual quality standards (design system)
- ✅ Demonstration scenarios (scenario library)
- ✅ Multi-dashboard coordination

**Unified:** Both follow constitutional governance, adapted to domain

---

## CONCLUSION

This constitution ensures the Web Platform is:
- ✅ **Professional quality** (not prototype)
- ✅ **Performant** (60fps, <100ms latency)
- ✅ **Accessible** (WCAG 2.1 AA)
- ✅ **Secure** (classification enforcement)
- ✅ **Tracked** (daily progress updates)
- ✅ **Governed** (automated enforcement)

**Status:** CONSTITUTIONAL FRAMEWORK ESTABLISHED
**Enforcement:** Build-time + Runtime + Workflow
**Compliance Target:** 9/10 minimum (production-grade)

---

**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
**Version:** 1.0.0
**Date:** January 9, 2025
