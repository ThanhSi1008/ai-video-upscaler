---
name: local-web-guidance
description: >-
  Local HTML/CSS/JS best-practice guides for web/static/ monitoring dashboard UI.
  Trigger when editing web/static/ (CSS layout, dark mode, responsive tables, dialogs,
  tabs, progress indicators, forms/sliders/switches, accessibility, HTML sanitization,
  UI responsiveness, or webmcp). Do NOT trigger for Python backend, CLI, or tests.
---

# Modern Web Guidance (Local Dashboard Curated Subset)

Use the curated local guides under `guides/` before modifying Vanilla HTML/CSS/JS in `web/static/`. All 39 retained guides are offline-ready and scoped to the local FastAPI monitoring UI (CSP `sha256`, zero `innerHTML`, `oklch()` + `light-dark()`).

## Curated Local Guide Index

- **HTML & Accessibility:** [HTML](guides/html/html.md) | [Accessibility](guides/accessibility/accessibility.md) | [Error Announcements](guides/accessibility/accessible-error-announcement.md)
- **CSS Layout & Responsiveness:** [CSS Core](guides/css/css.md) | [Layout](guides/css/css-layout.md) | [Responsive Design](guides/css/responsive-design.md) | [Design Token Reactivity](guides/css/design-token-reactivity.md) | [Overflow Clipping](guides/css/overflow-clipping-control.md)
- **Color & Dark Mode:** [Color](guides/visual-design/color.md) | [Dark Mode](guides/visual-design/dark-mode.md) | [Contrast Color](guides/visual-design/contrast-color.md) | [Scrollbars](guides/visual-design/customize-scrollbar-color-and-thickness.md) | [Dark Mode Toggle](guides/ui-components/dark-mode-toggle.md) | [Component Light/Dark Theme](guides/ui-atoms/component-specific-light-dark-theme.md)
- **Tables, Dialogs, Tabs & Controls:** [Responsive Table](guides/ui-atoms/responsive-table.md) | [Sticky Headers](guides/ui-atoms/state-aware-sticky-headers.md) | [Sortable Table](guides/ui-behaviors/sortable-table.md) | [Dialog](guides/ui-components/dialog.md) | [Light Dismiss Dialog](guides/ui-behaviors/light-dismiss-a-dialog.md) | [Prevent Modal Scroll](guides/ui-behaviors/prevent-page-scroll-under-modal-dialog.md) | [Declarative Dialog/Popover](guides/ui-behaviors/declarative-dialog-popover-control.md) | [Tabs](guides/ui-components/tabs.md) | [Card](guides/ui-components/card.md) | [Progress](guides/ui-components/progress.md) | [Spinner](guides/ui-components/spinner.md) | [Copy Button](guides/ui-components/copy-button.md)
- **Forms & Inputs:** [Forms](guides/forms/forms.md) | [Required Feedback](guides/forms/required-field-feedback.md) | [Validate After Interaction](guides/forms/validate-input-after-interaction.md) | [Slider](guides/forms/slider.md) | [Switch](guides/forms/switch.md)
- **Performance, Security & WebMCP:** [Performance](guides/performance/performance.md) | [Break Up Long Tasks](guides/performance/break-up-long-tasks.md) | [Defer Heavy Content](guides/performance/defer-rendering-heavy-content.md) | [Sanitize Untrusted HTML](guides/security/sanitize-untrusted-html.md) | [Local Network Access](guides/security/local-network-access.md) | [WebMCP](guides/webmcp/webmcp.md) | [Agentic Forms](guides/webmcp/agentic-forms.md) | [Agentic JS Tools](guides/webmcp/agentic-javascript-tools.md)

## Local Compliance Notes
- Verify changes against browser support targets (`Baseline Widely available` by default; check [GEMINI.md](../../../GEMINI.md) for repo invariants).
- Preserve strict CSP (`script-src 'self' 'sha256-...'`) and DOM API rendering (`textContent`/`createElement`, zero `innerHTML`).
