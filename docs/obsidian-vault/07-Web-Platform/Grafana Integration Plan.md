# Grafana Integration with Web Platform

**Status**: Planning
**Related**: [[Grafana Dashboard Setup]], [[Web Platform Master Plan]], [[Dashboard Coverage Matrix]]
**Date**: 2025-10-10

## Overview

This document outlines how the Grafana monitoring dashboard integrates with the PRISM-AI Web Platform, specifically supporting **Dashboard #4: System Internals & Data Lifecycle**.

## Current State

### Grafana Dashboard (Completed ✅)
- **Status**: Operational at http://localhost:3000
- **Components**: Grafana + Prometheus + Node Exporter
- **Metrics**: 15+ PRISM-AI metrics defined
- **Panels**: 6 visualization panels
- **Purpose**: Ops monitoring and development

### Web Platform (Planned)
- **Status**: Ready to implement
- **Timeline**: 11 weeks (or 6 weeks accelerated)
- **Dashboard #4**: System Internals & Data Lifecycle
- **Purpose**: Stakeholder demonstrations

## Integration Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   PRISM-AI Application                    │
│           (Optimization, GPU Processing, etc.)            │
└─────────────────┬───────────────────────┬────────────────┘
                  │                       │
                  │ Metrics Export        │ WebSocket Events
                  ↓                       ↓
    ┌─────────────────────┐    ┌──────────────────────┐
    │   Prometheus        │    │   Rust Backend       │
    │   (Time-series DB)  │    │   (Actix-web)        │
    └─────────┬───────────┘    └──────────┬───────────┘
              │                            │
              │ Queries                    │ WebSocket
              ↓                            ↓
    ┌─────────────────────┐    ┌──────────────────────┐
    │     Grafana         │    │   React Frontend     │
    │  (Ops Dashboard)    │    │  (Demo Dashboard)    │
    │  localhost:3000     │    │  Dashboard #4        │
    └─────────────────────┘    └──────────────────────┘
         For Operators              For Stakeholders
```

## Dashboard #4: System Internals & Data Lifecycle

From [[Dashboard Coverage Matrix]], Dashboard #4 requires:

### Core Features
1. **8-Phase Pipeline Visualization**
   - Phase transitions
   - Data flow tracking
   - Processing stages

2. **Performance Metrics**
   - Throughput measurement
   - Latency tracking
   - Resource utilization

3. **Constitutional Compliance**
   - Policy adherence
   - Governance tracking
   - Audit trails

4. **System Health**
   - Component status
   - Error monitoring
   - Recovery tracking

### Technology Stack (from Master Plan)
- **Frontend**: React 18 + TypeScript
- **Visualization**: ECharts, D3.js
- **Backend**: Rust + Actix-web
- **Real-time**: WebSocket
- **Styling**: Material-UI

## Integration Options

### Option 1: Parallel Systems (Current Recommended)

**Grafana**: Operational monitoring
- Use: Development, debugging, ops
- Audience: Engineers, operators
- Focus: Real-time metrics, alerting

**React Dashboard**: Demonstrations
- Use: Stakeholder presentations, demos
- Audience: DoD, investors, partners
- Focus: Polished UI, narrative flow

**Pros**: Clean separation, optimized for each use case
**Cons**: Duplicate visualization logic
**Effort**: Medium (build React dashboard from scratch)

### Option 2: Grafana Embedding

Embed Grafana panels in React dashboard using iframe.

**Pros**: Reuse existing visualizations
**Cons**: Limited customization, branding challenges
**Effort**: Low
**Recommendation**: ⚠️ Not recommended for stakeholder demos

### Option 3: Unified Backend

Share Prometheus metrics via WebSocket bridge.

```rust
// Rust backend queries Prometheus and streams to frontend
async fn metrics_websocket(
    req: HttpRequest,
    stream: web::Payload,
) -> Result<HttpResponse, Error> {
    // Query Prometheus
    let metrics = prometheus_client.query(...).await?;

    // Stream to WebSocket clients
    ws::start(MetricsActor::new(metrics), &req, stream)
}
```

**Pros**: Single source of truth, consistent data
**Cons**: Additional backend complexity
**Effort**: Medium
**Recommendation**: ✅ Best long-term solution

### Option 4: Separate Data Paths

Grafana and React dashboard independently collect metrics.

**Pros**: Complete independence
**Cons**: Potential data inconsistencies
**Effort**: High (duplicate instrumentation)
**Recommendation**: ⚠️ Only if requirements diverge significantly

## Recommended Implementation Plan

### Phase 1: Parallel Systems (Week 1-2)
**Goal**: Get both systems running independently

1. Keep Grafana for ops monitoring ✅ (Already done)
2. Build React Dashboard #4 with mock data
3. Implement WebSocket infrastructure in Rust backend
4. Create basic System Internals visualizations

**Deliverables**:
- [ ] React app scaffolding
- [ ] WebSocket server in Actix-web
- [ ] Mock data generator
- [ ] 2-3 basic panels working

### Phase 2: Shared Backend (Week 3-4)
**Goal**: Connect React dashboard to real metrics

1. Create Prometheus query wrapper in Rust
2. Stream metrics via WebSocket to React
3. Implement ECharts visualizations
4. Add real-time updates

**Deliverables**:
- [ ] Prometheus-to-WebSocket bridge
- [ ] React metrics consumers
- [ ] 4-6 panels with real data
- [ ] <100ms latency achieved

### Phase 3: Enhanced Visualizations (Week 5-6)
**Goal**: Polish for stakeholder demos

1. 8-phase pipeline visualization
2. Constitutional compliance tracking
3. Interactive exploration features
4. Responsive design + animations

**Deliverables**:
- [ ] All 6 tasks from [[Dashboard Coverage Matrix]] complete
- [ ] 4 demonstration scenarios working
- [ ] Stakeholder-ready polish

### Phase 4: Production Ready (Week 7-8)
**Goal**: Deploy and harden

1. Authentication/authorization
2. HTTPS/TLS configuration
3. Load testing (60fps target)
4. Monitoring and alerting

**Deliverables**:
- [ ] Production deployment
- [ ] Security audit passed
- [ ] Performance targets met
- [ ] Documentation complete

## Metrics Mapping

### Current Grafana Metrics → Dashboard #4 Panels

| Grafana Metric | Dashboard #4 Panel |
|----------------|-------------------|
| `prism_ai_optimization_iterations` | Pipeline Throughput |
| `prism_ai_solution_quality` | Performance Metrics |
| `prism_ai_gpu_utilization` | Resource Utilization |
| `prism_ai_processing_time_seconds` | Latency Distribution |
| `prism_ai_active_problems` | System Load |
| `prism_ai_errors_total` | Health Status |

### New Metrics Needed for Dashboard #4

```rust
// Phase tracking
pub static ref CURRENT_PHASE: IntGauge = register_int_gauge!(
    "prism_ai_current_phase",
    "Current processing phase (1-8)"
).unwrap();

// Constitutional compliance
pub static ref COMPLIANCE_SCORE: Gauge = register_gauge!(
    "prism_ai_compliance_score",
    "Constitutional compliance score (0-100)"
).unwrap();

// Pipeline metrics
pub static ref PIPELINE_BACKLOG: IntGauge = register_int_gauge!(
    "prism_ai_pipeline_backlog",
    "Number of items in processing queue"
).unwrap();
```

## Code Integration Points

### Backend (Rust + Actix-web)

```rust
// src/web_platform/metrics_api.rs
use actix_web::{web, HttpResponse};
use prometheus::TextEncoder;

pub async fn metrics_endpoint() -> HttpResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(buffer)
}

// src/web_platform/websocket.rs
use actix_web_actors::ws;

pub struct MetricsWebSocket {
    prometheus_client: PrometheusClient,
}

impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Start periodic metrics push
        ctx.run_interval(Duration::from_secs(1), |act, ctx| {
            let metrics = act.prometheus_client.query_all();
            ctx.text(serde_json::to_string(&metrics).unwrap());
        });
    }
}
```

### Frontend (React + TypeScript)

```typescript
// src/hooks/useMetrics.ts
import { useEffect, useState } from 'react';

export function useMetrics() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080/ws/metrics');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMetrics(data);
    };

    return () => ws.close();
  }, []);

  return metrics;
}

// src/components/SystemInternals/PipelineVisualization.tsx
import { useMetrics } from '../../hooks/useMetrics';
import { LineChart } from '@mui/x-charts';

export function PipelineVisualization() {
  const metrics = useMetrics();

  return (
    <LineChart
      series={[{
        data: metrics?.optimizationIterations || [],
        label: 'Iterations/sec'
      }]}
    />
  );
}
```

## Success Criteria

### Technical Targets (from [[Web Platform Master Plan]])
- [x] Grafana operational
- [ ] React dashboard rendering at 60fps
- [ ] WebSocket latency <100ms
- [ ] Test coverage >80%
- [ ] All 6 Dashboard #4 tasks complete

### Demonstration Scenarios (from [[Dashboard Coverage Matrix]])

**Scenario 1**: Real-time Pipeline Monitoring
- Show 8-phase pipeline processing graph coloring problem
- Display phase transitions in real-time
- Highlight resource utilization per phase

**Scenario 2**: Performance Analysis
- Load historical optimization runs
- Compare performance across algorithms
- Show improvement over time

**Scenario 3**: System Health Dashboard
- Display all component health statuses
- Show error rates and recovery times
- Demonstrate alerting capabilities

**Scenario 4**: Constitutional Compliance
- Track policy adherence in real-time
- Show governance audit trail
- Demonstrate automated compliance checks

## Timeline Integration

### Web Platform Timeline (from Master Plan)
- **Weeks 1-2**: Frontend scaffolding
- **Weeks 3-4**: Backend WebSocket
- **Weeks 5-6**: Dashboard #4 core features
- **Weeks 7-8**: Dashboard #4 polish
- **Weeks 9-10**: Testing and optimization
- **Week 11**: Documentation and handoff

### Grafana Enhancement Timeline (Parallel)
- **Week 1**: Add phase tracking metrics
- **Week 2**: Add constitutional compliance metrics
- **Week 3**: Create HFT-specific dashboard
- **Week 4**: Create materials science dashboard

## Related Documentation

- [[04-Development/Grafana Dashboard Setup]] - Current implementation
- [[05-Status/Grafana Dashboard Status]] - Operational status
- [[Web Platform Master Plan]] - Overall web platform strategy
- [[Dashboard Coverage Matrix]] - Dashboard #4 specifications
- [[Web Platform Overview]] - Project overview and structure

## Next Actions

### Immediate (This Week)
- [ ] Review [[Web Platform Master Plan]] with team
- [ ] Decide on integration approach (Option 1 or 3 recommended)
- [ ] Set up React project skeleton
- [ ] Create initial WebSocket prototype

### Short Term (Next 2 Weeks)
- [ ] Implement Prometheus-to-WebSocket bridge
- [ ] Build first Dashboard #4 panel
- [ ] Add phase tracking to PRISM-AI code
- [ ] Create mock data for development

### Medium Term (Month 1-2)
- [ ] Complete all 6 Dashboard #4 tasks
- [ ] Implement 4 demonstration scenarios
- [ ] Polish UI for stakeholder presentations
- [ ] Load testing and optimization

## Questions to Resolve

1. **Timing**: Build web platform now or after SBIR submission?
2. **Scope**: All 4 dashboards or just Dashboard #4 first?
3. **Resources**: Who will develop React frontend?
4. **Infrastructure**: Where to deploy (AWS, GCP, on-prem)?
5. **Integration**: Option 1 (parallel) or Option 3 (unified backend)?

---

**Status**: 📋 Planning Phase
**Dependencies**: Grafana dashboard (✅ complete), Web platform decision
**Owner**: TBD
**Target Start**: TBD (pending SBIR timing decision)
