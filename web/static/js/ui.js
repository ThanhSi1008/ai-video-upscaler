/**
 * web/static/js/ui.js
 * Core accessible UI primitives & safe DOM helpers:
 * - Zero innerHTML / dangerous DOM sinks (all nodes built via createElement + textContent).
 * - Accessible Tabs (focusgroup="tablist nomemory", hidden="until-found", beforematch, roving fallback).
 * - Light-dismiss Modal <dialog> fallback for browsers without HTMLDialogElement.prototype.closedBy.
 * - Persistent Toast Notifications (popover="manual", modal suppression queue, ariaNotify + live-region).
 * - Tri-state Color Scheme controller (<meta name="color-scheme"> + localStorage + cross-tab sync).
 */

let toastSeq = 0;
const pendingToastQueue = [];

export function clearChildren(node) {
  if (node) {
    node.replaceChildren();
  }
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs || {})) {
    if (val === undefined || val === null) continue;
    if (key === "className") {
      node.className = String(val);
    } else if (key === "textContent") {
      node.textContent = String(val);
    } else if (key === "hidden") {
      if (val === "until-found") {
        node.setAttribute("hidden", "until-found");
      } else {
        node.hidden = Boolean(val);
      }
    } else if (key === "disabled" || key === "checked" || key === "selected" || key === "readOnly") {
      node[key] = Boolean(val);
    } else if (key === "value") {
      node.value = String(val);
    } else if (key === "dataset" && typeof val === "object") {
      for (const [dk, dv] of Object.entries(val)) {
        if (dv !== undefined && dv !== null) {
          node.dataset[dk] = String(dv);
        }
      }
    } else if (key.startsWith("on") && typeof val === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), val);
    } else {
      node.setAttribute(key, String(val));
    }
  }

  const childList = Array.isArray(children) ? children : [children];
  for (const child of childList) {
    if (child === undefined || child === null || child === false) continue;
    if (child instanceof Node) {
      node.append(child);
    } else {
      node.append(document.createTextNode(String(child)));
    }
  }
  return node;
}

export async function apiFetch(url, options = {}) {
  const fetchOpts = { ...options };
  const headers = new Headers(fetchOpts.headers || {});
  headers.set("Accept", "application/json");

  if (
    fetchOpts.body !== undefined &&
    fetchOpts.body !== null &&
    typeof fetchOpts.body === "object" &&
    !(fetchOpts.body instanceof FormData)
  ) {
    headers.set("Content-Type", "application/json");
    fetchOpts.body = JSON.stringify(fetchOpts.body);
  }
  fetchOpts.headers = headers;

  const response = await fetch(url, fetchOpts);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => ({}))
    : await response.text().catch(() => "");

  if (!response.ok) {
    const rawDetail =
      typeof payload === "object" && payload !== null
        ? payload.detail || payload.message || JSON.stringify(payload)
        : String(payload || `HTTP ${response.status}`);
    const detail = typeof rawDetail === "string" ? rawDetail : JSON.stringify(rawDetail);
    const err = new Error(detail);
    err.status = response.status;
    err.payload = typeof payload === "object" && payload !== null ? payload : null;
    throw err;
  }
  return payload;
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return "--:--";
  }
  const total = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function isAnyModalDialogOpen() {
  for (const dlg of document.querySelectorAll("dialog")) {
    if (dlg.open) return true;
  }
  return false;
}

function announceScreenReader(text) {
  if (typeof document.ariaNotify === "function") {
    try {
      document.ariaNotify(text);
      return;
    } catch {
      // Fall back to live region below
    }
  }
  const liveRegion = document.getElementById("sr-live-region");
  if (liveRegion) {
    liveRegion.textContent = "";
    window.setTimeout(() => {
      liveRegion.textContent = text;
    }, 30);
  }
}

function flushQueuedToasts() {
  if (isAnyModalDialogOpen()) return;
  while (pendingToastQueue.length > 0) {
    const item = pendingToastQueue.shift();
    renderToastNow(item.message, item.tone, item.durationMs);
  }
}

function relayoutToastStack() {
  const region = document.getElementById("toast-region");
  if (!region) return;
  const items = Array.from(region.querySelectorAll(".toast-item"));
  if (items.length === 0) return;

  const vh = Math.max(120, Number(window.innerHeight || 720));
  const edgePx = vh < 360 ? 8 : 16;
  const gapPx = vh < 360 ? 6 : 9;
  const availH = Math.max(88, vh - edgePx * 2 - gapPx * Math.max(0, items.length - 1));

  // Measure natural heights first, then cap per-toast height only when stack exceeds viewport (WCAG 1.4.10 Reflow)
  const natHeights = items.map((item) => {
    item.style.setProperty("--toast-edge", `${edgePx}px`);
    item.style.removeProperty("--toast-max-h");
    const rect = item.getBoundingClientRect();
    return rect.height || item.offsetHeight || 48;
  });

  const totalNat = natHeights.reduce((acc, h) => acc + h, 0);
  if (totalNat > availH) {
    const slotCap = Math.max(36, Math.floor(availH / items.length));
    for (const item of items) {
      item.style.setProperty("--toast-max-h", `${slotCap}px`);
    }
  }

  let offsetPx = 0;
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    item.style.setProperty("--toast-offset", `${Math.round(offsetPx)}px`);
    const rect = item.getBoundingClientRect();
    const h = rect.height || item.offsetHeight || 48;
    offsetPx += h + gapPx;
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("resize", relayoutToastStack, { passive: true });
}

function renderToastNow(message, tone, durationMs) {
  const region = document.getElementById("toast-region");
  if (!region) return;

  const priorFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const toastId = `studio-toast-${++toastSeq}`;
  const toast = document.createElement("div");
  toast.id = toastId;
  toast.className = "toast-item";
  toast.dataset.tone = tone;
  toast.setAttribute("popover", "manual");

  const msgSpan = el("span", { textContent: message });
  const closeBtn = el("button", {
    type: "button",
    className: "btn btn-ghost",
    "aria-label": "Đóng thông báo",
    textContent: "✕",
  });
  closeBtn.setAttribute("popovertarget", toastId);
  closeBtn.setAttribute("popovertargetaction", "hide");

  let dismissTimer = null;
  let removed = false;

  const dismissToast = (manualByKeyboard = false) => {
    if (removed) return;
    removed = true;
    if (dismissTimer) {
      window.clearTimeout(dismissTimer);
      dismissTimer = null;
    }
    const hadFocus = toast.contains(document.activeElement);
    try {
      if (typeof toast.hidePopover === "function") {
        toast.hidePopover();
      }
    } catch {
      // Ignore if already hidden
    }
    toast.classList.remove(":popover-open");
    toast.remove();
    relayoutToastStack();

    if (manualByKeyboard || hadFocus) {
      const nextToastBtn = region.querySelector(".toast-item button");
      if (nextToastBtn instanceof HTMLElement) {
        nextToastBtn.focus();
      } else if (priorFocus && priorFocus.isConnected) {
        priorFocus.focus();
      }
    }
  };

  const scheduleAutoDismiss = () => {
    if (durationMs <= 0 || removed) return;
    if (dismissTimer) window.clearTimeout(dismissTimer);
    dismissTimer = window.setTimeout(() => {
      if (toast.matches(":focus-within")) {
        return;
      }
      dismissToast(false);
    }, durationMs);
  };

  toast.addEventListener("focusin", () => {
    if (dismissTimer) {
      window.clearTimeout(dismissTimer);
      dismissTimer = null;
    }
  });
  toast.addEventListener("focusout", (ev) => {
    if (!toast.contains(ev.relatedTarget)) {
      scheduleAutoDismiss();
    }
  });
  toast.addEventListener("toggle", (ev) => {
    if (ev.newState === "closed" && !removed) {
      dismissToast(false);
    }
  });
  closeBtn.addEventListener("click", (ev) => {
    ev.preventDefault();
    dismissToast(ev.detail === 0);
  });

  toast._dismissToast = dismissToast;
  toast.append(msgSpan, closeBtn);
  region.append(toast);

  // Keep stack tidy (max 4 concurrent toasts) with clean timer/popover teardown
  const existing = region.querySelectorAll(".toast-item");
  if (existing.length > 4) {
    if (typeof existing[0]._dismissToast === "function") {
      existing[0]._dismissToast(false);
    } else {
      existing[0].remove();
    }
  }

  if (typeof toast.showPopover === "function") {
    try {
      toast.showPopover();
    } catch {
      toast.classList.add(":popover-open");
    }
  } else {
    toast.classList.add(":popover-open");
  }
  relayoutToastStack();

  announceScreenReader(message);
  scheduleAutoDismiss();
}

export function showToast(message, tone = "info", durationMs = 5200) {
  if (!message) return;
  if (isAnyModalDialogOpen()) {
    pendingToastQueue.push({ message, tone, durationMs });
    return;
  }
  renderToastNow(message, tone, durationMs);
}

export function initDialogLightDismiss(dialog) {
  if (!(dialog instanceof HTMLDialogElement)) return;

  // Flush suppressed background toasts once any modal dialog closes
  dialog.addEventListener("close", () => {
    window.setTimeout(flushQueuedToasts, 40);
  });

  // Fallback for browsers without declarative <dialog closedby="any">
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const isOutside = (ev) => {
      const rect = dialog.getBoundingClientRect();
      return (
        ev.clientX < rect.left ||
        ev.clientX > rect.right ||
        ev.clientY < rect.top ||
        ev.clientY > rect.bottom
      );
    };
    let pressedOutside = false;
    dialog.addEventListener("pointerdown", (ev) => {
      if (!dialog.open || ev.button !== 0) {
        pressedOutside = false;
        return;
      }
      pressedOutside = isOutside(ev);
    });
    dialog.addEventListener("pointerup", (ev) => {
      if (!dialog.open || ev.button !== 0) {
        pressedOutside = false;
        return;
      }
      if (pressedOutside && isOutside(ev)) {
        dialog.close();
      }
      pressedOutside = false;
    });
    dialog.addEventListener("pointercancel", () => {
      pressedOutside = false;
    });
  }
}

export function initThemeSwitcher(onThemeChange) {
  const meta = document.querySelector('meta[name="color-scheme"]');
  const select = document.getElementById("theme-select");
  if (!meta || !(select instanceof HTMLSelectElement)) return;

  const applyScheme = (rawScheme, persist = false) => {
    const normalized =
      rawScheme === "dark" || rawScheme === "light" ? rawScheme : "light dark";
    meta.content = normalized;
    select.value = normalized;

    if (persist) {
      try {
        if (normalized === "light dark") {
          localStorage.removeItem("color-scheme");
        } else {
          localStorage.setItem("color-scheme", normalized);
        }
      } catch {
        // Storage blocked; theme still applies via <meta>
      }
    }
    if (typeof onThemeChange === "function") {
      window.requestAnimationFrame(() => onThemeChange(normalized));
    }
  };

  let initial = "light dark";
  try {
    initial = localStorage.getItem("color-scheme") ?? "light dark";
  } catch {
    initial = meta.content || "light dark";
  }
  applyScheme(initial, false);

  select.addEventListener("change", () => {
    applyScheme(select.value, true);
  });

  window.addEventListener("storage", (ev) => {
    if (ev.key === "color-scheme") {
      applyScheme(ev.newValue ?? "light dark", false);
    }
  });

  const mediaDark = window.matchMedia?.("(prefers-color-scheme: dark)");
  mediaDark?.addEventListener?.("change", () => {
    if (typeof onThemeChange === "function") {
      window.requestAnimationFrame(() => onThemeChange(meta.content));
    }
  });
}

export function initTabs(onTabChange) {
  const tablist = document.querySelector('[role="tablist"]');
  const panelsStack = document.querySelector(".tabpanels-stack");
  if (!tablist || !panelsStack) {
    return { selectTabById: () => {} };
  }

  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  const panels = Array.from(panelsStack.querySelectorAll('[role="tabpanel"]'));
  const hasNativeFocusGroup = Object.hasOwn(HTMLElement.prototype, "focusGroup");
  const supportsUntilFound = "onbeforematch" in HTMLElement.prototype;
  let initialized = false;

  const panelForTab = (tab) =>
    panels.find((p) => p.getAttribute("aria-labelledby") === tab.id) || null;
  const tabForPanel = (panel) =>
    tabs.find((t) => t.id === panel.getAttribute("aria-labelledby")) || null;

  const scrollTabIntoView = (tab) => {
    try {
      tab.scrollIntoView({
        block: "nearest",
        inline: "nearest",
        container: "nearest",
      });
    } catch {
      tab.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  };

  const selectTab = (selectedTab, { focusTab = false, fromBeforeMatch = false } = {}) => {
    if (!selectedTab) return;
    const targetPanel = panelForTab(selectedTab);
    const alreadySelected =
      initialized &&
      selectedTab.getAttribute("aria-selected") === "true" &&
      targetPanel &&
      !targetPanel.hasAttribute("hidden");
    if (alreadySelected) {
      if (focusTab) {
        selectedTab.focus();
      }
      scrollTabIntoView(selectedTab);
      return;
    }

    const activeEl = document.activeElement;
    const hidingFocusedPanel =
      !focusTab &&
      !fromBeforeMatch &&
      activeEl instanceof Node &&
      panels.some((p) => p !== targetPanel && p.contains(activeEl));

    for (const tab of tabs) {
      const isSelected = tab === selectedTab;
      tab.setAttribute("aria-selected", String(isSelected));
      tab.toggleAttribute("focusgroupstart", isSelected);
      if (!hasNativeFocusGroup) {
        tab.tabIndex = isSelected ? 0 : -1;
      }
      const panel = panelForTab(tab);
      if (panel) {
        if (isSelected) {
          panel.hidden = false;
          panel.removeAttribute("hidden");
        } else if (supportsUntilFound) {
          panel.setAttribute("hidden", "until-found");
        } else {
          panel.hidden = true;
        }
      }
    }

    scrollTabIntoView(selectedTab);

    if (focusTab) {
      selectedTab.focus();
    } else if (hidingFocusedPanel) {
      const targetHeading = targetPanel?.querySelector('h2[tabindex="-1"], [tabindex="-1"]');
      if (targetHeading instanceof HTMLElement) {
        targetHeading.focus();
      } else {
        selectedTab.focus();
      }
    }

    if (initialized && typeof onTabChange === "function") {
      onTabChange(selectedTab.id);
    }
  };

  // Initialize ARIA + roving fallback state
  const currentSelected =
    tabs.find((t) => t.getAttribute("aria-selected") === "true") || tabs[0];
  selectTab(currentSelected, { focusTab: false });
  initialized = true;

  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab, { focusTab: false }));
    if (hasNativeFocusGroup) {
      tab.addEventListener("focus", () => {
        if (tab.getAttribute("aria-selected") !== "true") {
          selectTab(tab, { focusTab: false });
        }
      });
    }
  }

  // Roving tabindex keyboard navigation fallback when native focusgroup is unavailable
  if (!hasNativeFocusGroup) {
    tablist.addEventListener("keydown", (ev) => {
      const idx = tabs.indexOf(document.activeElement);
      if (idx < 0) return;
      let nextIdx = -1;
      if (ev.key === "ArrowRight") {
        nextIdx = (idx + 1) % tabs.length;
      } else if (ev.key === "ArrowLeft") {
        nextIdx = (idx - 1 + tabs.length) % tabs.length;
      } else if (ev.key === "Home") {
        nextIdx = 0;
      } else if (ev.key === "End") {
        nextIdx = tabs.length - 1;
      }
      if (nextIdx >= 0) {
        ev.preventDefault();
        selectTab(tabs[nextIdx], { focusTab: true });
      }
    });
  }

  // Synchronize tab selection when Find-in-Page matches text inside hidden="until-found" panel
  panelsStack.addEventListener("beforematch", (ev) => {
    const matchedPanel =
      ev.target instanceof Element ? ev.target.closest('[role="tabpanel"]') : null;
    if (!matchedPanel) return;
    const ownerTab = tabForPanel(matchedPanel);
    if (ownerTab) {
      selectTab(ownerTab, { focusTab: false, fromBeforeMatch: true });
    }
  });

  // Scroll-driven edge fades fallback via IntersectionObserver
  if (
    typeof CSS !== "undefined" &&
    typeof CSS.supports === "function" &&
    !CSS.supports("animation-timeline", "scroll()") &&
    typeof IntersectionObserver === "function"
  ) {
    for (const scroller of document.querySelectorAll(".scroller")) {
      const first = scroller.firstElementChild;
      const last = scroller.lastElementChild;
      if (!first || !last) continue;
      const observer = new IntersectionObserver(
        (entries) => {
          for (const { target, intersectionRatio } of entries) {
            scroller.classList.toggle(
              target === first ? "more-start" : "more-end",
              intersectionRatio < 1
            );
          }
        },
        { root: scroller, threshold: 1, rootMargin: "1px" }
      );
      observer.observe(first);
      observer.observe(last);
    }
  }

  return {
    selectTabById(tabId, opts = {}) {
      const target = tabs.find((t) => t.id === tabId);
      if (target) {
        selectTab(target, opts);
      }
    },
  };
}
