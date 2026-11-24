/**
 * web/static/js/fs_picker.js
 * Safe Local Disk & Folder Browser Modal (<dialog closedby="any">):
 * - Supports multi-file + whole-folder selection for Queue, or single-file selection for Compare/Subtrans.
 * - Displays natural episode order, shows.yaml routing badges, drive free space, and already-upscaled hints.
 */

import { apiFetch, clearChildren, el, initDialogLightDismiss } from "./ui.js";

let initialized = false;
let activeSession = null;
let lastVisitedPath = "";

const _QUOTE_PAIRS = {
  '"': '"',
  "'": "'",
  "\u201c": "\u201d",
  "\u201d": "\u201c",
  "\u2018": "\u2019",
  "\u2019": "\u2018",
};

export const stripOuterQuotes = (raw) => {
  let s = String(raw || "").trim();
  while (s.length >= 2) {
    const first = s[0];
    const last = s[s.length - 1];
    if ((first === last && (first === '"' || first === "'")) || _QUOTE_PAIRS[first] === last) {
      s = s.slice(1, -1).trim();
    } else {
      break;
    }
  }
  return s;
};

export function initFsPickerModal() {
  if (initialized) return;
  const dialog = document.getElementById("fs-picker-dialog");
  if (!(dialog instanceof HTMLDialogElement)) return;
  initialized = true;

  initDialogLightDismiss(dialog);

  const btnDrives = document.getElementById("fs-btn-drives");
  const btnWorkspace = document.getElementById("fs-btn-workspace");
  const btnUp = document.getElementById("fs-btn-up");
  const btnPickFolder = document.getElementById("fs-btn-pick-folder");
  const jumpInput = document.getElementById("fs-jump-input");
  const btnJump = document.getElementById("fs-btn-jump");
  const filterInput = document.getElementById("fs-filter-input");

  btnDrives?.addEventListener("click", () => {
    activeSession?.browse("drives");
  });

  btnWorkspace?.addEventListener("click", () => {
    activeSession?.browse(activeSession.workspaceRoot || "");
  });

  btnUp?.addEventListener("click", () => {
    if (activeSession?.parentPath) {
      activeSession.browse(activeSession.parentPath);
    } else {
      activeSession?.browse("drives");
    }
  });

  const triggerJump = () => {
    if (!activeSession || !(jumpInput instanceof HTMLInputElement)) return;
    const target = stripOuterQuotes(jumpInput.value);
    activeSession.browse(target || "drives");
  };

  btnJump?.addEventListener("click", triggerJump);
  jumpInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      if (ev.isComposing || ev.keyCode === 229) return;
      ev.preventDefault();
      triggerJump();
    }
  });

  filterInput?.addEventListener("input", () => {
    activeSession?.renderEntries();
  });

  btnPickFolder?.addEventListener("click", () => {
    if (!activeSession || !activeSession.currentPath) return;
    const chosen = [activeSession.currentPath];
    activeSession.resolved = true;
    dialog.close("folder");
    activeSession.resolve(chosen);
  });

  dialog.addEventListener("close", () => {
    const outcome = dialog.returnValue;
    dialog.returnValue = "";
    document.getElementById("fs-entries-list")?.removeAttribute("aria-busy");
    if (dialog.open || !activeSession) return;
    const sess = activeSession;
    activeSession = null;
    if (sess.resolved) return;
    if (outcome === "confirm") {
      sess.resolve(Array.from(sess.selected));
    } else {
      sess.resolve([]);
    }
  });
}

export function openFsPicker(options = {}) {
  initFsPickerModal();
  const dialog = document.getElementById("fs-picker-dialog");
  if (!(dialog instanceof HTMLDialogElement)) {
    return Promise.resolve([]);
  }

  const titleEl = document.getElementById("fs-picker-title");
  const pathPill = document.getElementById("fs-current-path");
  const folderBanner = document.getElementById("fs-folder-banner");
  const folderSummary = document.getElementById("fs-folder-summary");
  const entriesList = document.getElementById("fs-entries-list");
  const selectionCount = document.getElementById("fs-selection-count");
  const btnUp = document.getElementById("fs-btn-up");
  const btnConfirm = document.getElementById("fs-btn-confirm");
  const btnPickFolder = document.getElementById("fs-btn-pick-folder");
  const jumpInput = document.getElementById("fs-jump-input");
  const filterInput = document.getElementById("fs-filter-input");

  const cfg = {
    title: options.title || "Chọn Video hoặc Thư mục trên Máy trạm",
    multi: Boolean(options.multi),
    allowFolders: Boolean(options.allowFolders),
    foldersOnly: Boolean(options.foldersOnly),
    includeSubtitles: Boolean(options.includeSubtitles),
    allow4k: Boolean(options.allow4k),
    initialPath: stripOuterQuotes(options.initialPath ?? (lastVisitedPath || "")),
  };

  if (titleEl) {
    titleEl.textContent = cfg.title;
  }
  if (filterInput instanceof HTMLInputElement) {
    filterInput.value = "";
  }

  return new Promise((resolve) => {
    const session = {
      selected: new Set(),
      currentPath: "",
      parentPath: null,
      workspaceRoot: "",
      lastEntries: [],
      browseSeq: 0,
      resolved: false,
      resolve,
      updateFooter() {
        const count = this.selected.size;
        if (selectionCount) {
          selectionCount.textContent =
            count > 0
              ? `Đã chọn ${count} mục.`
              : cfg.foldersOnly
                ? "Mở thư mục đích rồi nhấn “✓ Chọn thư mục này” (hoặc “+ Chọn thư mục” trên dòng con)."
                : cfg.multi
                  ? "Nhấp vào thư mục để mở, hoặc đánh dấu các tệp cần thêm."
                  : "Nhấp vào một tệp để chọn nhanh.";
        }
        if (btnConfirm instanceof HTMLButtonElement) {
          btnConfirm.disabled = count === 0;
        }
      },
      renderEntries() {
        if (!entriesList) return;
        clearChildren(entriesList);
        const filterKw =
          filterInput instanceof HTMLInputElement ? filterInput.value.trim().toLowerCase() : "";
        let renderedCount = 0;
        for (const item of this.lastEntries) {
          if (cfg.foldersOnly && item.type !== "drive" && item.type !== "dir") {
            continue;
          }
          if (item.type === "video_4k" && !cfg.allow4k && !cfg.multi) {
            continue;
          }
          if (filterKw && !String(item.name || "").toLowerCase().includes(filterKw)) {
            continue;
          }
          const row = this._renderEntryRow(item);
          if (row) {
            entriesList.append(row);
            renderedCount++;
          }
        }
        if (renderedCount === 0) {
          entriesList.append(
            el("p", {
              className: "empty-hint",
              textContent: filterKw
                ? `Không có mục nào khớp từ khóa "${filterKw}".`
                : cfg.foldersOnly
                  ? "Không có thư mục con (bạn vẫn có thể chọn chính thư mục đang mở ở thanh trên)."
                  : "Thư mục này chưa có thư mục con hoặc tệp media phù hợp.",
            })
          );
        }
      },
      async browse(targetPath, { allowFallback = false } = {}) {
        if (!entriesList || activeSession !== this || this.resolved) return;
        const cleanTarget = stripOuterQuotes(targetPath);
        const reqId = ++this.browseSeq;
        const isCurrent = () =>
          activeSession === this && !this.resolved && dialog.open && reqId === this.browseSeq;

        entriesList.setAttribute("aria-busy", "true");
        if (entriesList.childElementCount === 0) {
          entriesList.append(el("p", { className: "empty-hint", textContent: "Đang đọc thư mục…" }));
        }

        try {
          const qs = new URLSearchParams();
          if (cleanTarget) qs.set("path", cleanTarget);
          if (cfg.includeSubtitles) qs.set("include_subtitles", "true");
          const data = await apiFetch(`/api/fs/browse?${qs.toString()}`);
          if (!isCurrent()) return;

          this.currentPath = data.current_path || "";
          this.parentPath = data.parent_path || null;
          this.workspaceRoot = data.workspace_root || this.workspaceRoot;
          if (this.currentPath) {
            lastVisitedPath = this.currentPath;
          }

          if (pathPill) {
            pathPill.textContent = this.currentPath || "Danh sách Ổ đĩa Máy trạm";
          }
          if (jumpInput instanceof HTMLInputElement) {
            jumpInput.value = this.currentPath || "";
          }
          if (btnUp instanceof HTMLButtonElement) {
            btnUp.disabled = !this.currentPath;
          }

          // Folder whole-select banner
          if (folderBanner && folderSummary) {
            if (cfg.foldersOnly && this.currentPath) {
              folderSummary.textContent = `Thư mục đích hiện tại: ${this.currentPath}`;
              if (btnPickFolder) btnPickFolder.textContent = "✓ Chọn thư mục này";
              folderBanner.hidden = false;
            } else if (cfg.allowFolders && this.currentPath && Number(data.source_video_count || 0) > 0) {
              const routeModel = data.folder_routing?.model_id || "animevideov3";
              const routePol = data.folder_routing?.dedup_policy || "default";
              const mixNote = data.folder_routing?.mixed_routing ? " [hỗn hợp theo file]" : "";
              folderSummary.textContent = `Thư mục này có ${data.source_video_count} tập nguồn • Định tuyến: ${routeModel} (${routePol})${mixNote}`;
              if (btnPickFolder) btnPickFolder.textContent = "✓ Chọn cả thư mục này";
              folderBanner.hidden = false;
            } else {
              folderBanner.hidden = true;
            }
          }

          this.lastEntries = Array.isArray(data.entries) ? data.entries : [];
          this.renderEntries();
          this.updateFooter();
        } catch (err) {
          if (!isCurrent()) return;
          if (allowFallback && cleanTarget && cleanTarget.toLowerCase() !== "drives") {
            const trimmed = cleanTarget.replace(/[\\/]+$/, "");
            const sepIdx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
            const nextFallback =
              sepIdx > 0
                ? trimmed[sepIdx - 1] === ":"
                  ? trimmed.slice(0, sepIdx + 1)
                  : trimmed.slice(0, sepIdx)
                : "";
            if (nextFallback && nextFallback.toLowerCase() !== cleanTarget.toLowerCase()) {
              return this.browse(nextFallback, { allowFallback: true });
            }
            return this.browse("", { allowFallback: false });
          }
          for (const stale of entriesList.querySelectorAll(".fs-error-banner")) {
            stale.remove();
          }
          if (!entriesList.querySelector(".fs-entry-row")) {
            clearChildren(entriesList);
          }
          entriesList.prepend(
            el("div", {
              className: "fs-error-banner",
              role: "alert",
              textContent: `Không thể mở thư mục: ${err.message}`,
            })
          );
          entriesList.scrollTop = 0;
        } finally {
          if (isCurrent()) {
            entriesList.removeAttribute("aria-busy");
          }
        }
      },
      _renderEntryRow(item) {
        const row = el("div", { className: "fs-entry-row" });
        const main = el("div", { className: "fs-entry-main" });
        const rightCluster = el("div", { className: "button-cluster" });

        if (item.type === "drive" || item.type === "dir") {
          const icon = item.type === "drive" ? "🖴" : "📁";
          const openBtn = el(
            "button",
            {
              type: "button",
              className: "fs-open-dir-btn",
              onClick: (ev) => {
                ev.stopPropagation();
                this.browse(item.path);
              },
            },
            [`${icon} ${item.name}`]
          );
          main.append(openBtn);

          if (item.type === "drive") {
            rightCluster.append(
              el("span", {
                className: "badge",
                textContent: `Trống ${item.free_gb} / ${item.total_gb} GB`,
              })
            );
          } else if (cfg.allowFolders) {
            const addDirBtn = el("button", {
              type: "button",
              className: "btn btn-ghost",
              textContent: "+ Chọn thư mục",
              onClick: (ev) => {
                ev.stopPropagation();
                if (!cfg.multi) this.selected.clear();
                this.selected.add(item.path);
                this.resolved = true;
                dialog.close("confirm");
                this.resolve(Array.from(this.selected));
              },
            });
            rightCluster.append(addDirBtn);
          }
          row.append(main, rightCluster);
          return row;
        }

        // File entry (video, video_4k, subtitle)
        const is4kOutput = item.type === "video_4k";
        const selectable = !is4kOutput || cfg.allow4k;
        const icon = item.type === "subtitle" ? "💬" : is4kOutput ? "✨" : "🎞️";

        if (selectable) {
          const inputType = cfg.multi ? "checkbox" : "radio";
          const check = el("input", {
            type: inputType,
            name: "fs-picker-choice",
            checked: this.selected.has(item.path),
            "aria-label": `Chọn ${item.name}`,
            onClick: (ev) => {
              ev.stopPropagation();
            },
            onChange: (ev) => {
              if (!cfg.multi) {
                this.selected.clear();
                if (ev.target.checked) this.selected.add(item.path);
              } else if (ev.target.checked) {
                this.selected.add(item.path);
              } else {
                this.selected.delete(item.path);
              }
              this.updateFooter();
            },
          });

          const nameLabel = el(
            "button",
            {
              type: "button",
              className: "fs-open-dir-btn",
              onClick: (ev) => {
                ev.stopPropagation();
                if (!cfg.multi) {
                  this.selected.clear();
                  this.selected.add(item.path);
                  this.resolved = true;
                  dialog.close("confirm");
                  this.resolve([item.path]);
                } else {
                  check.checked = !check.checked;
                  check.dispatchEvent(new Event("change"));
                }
              },
            },
            [`${icon} ${item.name}`]
          );
          main.append(check, nameLabel);
        } else {
          main.append(
            el("span", {
              className: "muted-meta",
              textContent: `${icon} ${item.name} (Bản thành phẩm 4K)`,
            })
          );
        }

        if (item.routing?.model_id && !is4kOutput) {
          rightCluster.append(
            el("span", {
              className: "badge",
              textContent: item.routing.model_id,
            })
          );
        }
        if (item.already_upscaled) {
          rightCluster.append(
            el("span", {
              className: "badge badge-ok",
              textContent: "✓ Đã có 4K",
            })
          );
        }
        if (typeof item.size_mb === "number") {
          rightCluster.append(
            el("span", {
              className: "muted-meta",
              textContent: `${item.size_mb} MB`,
            })
          );
        }

        row.append(main, rightCluster);
        return row;
      },
    };

    if (activeSession && !activeSession.resolved) {
      activeSession.resolved = true;
      activeSession.resolve([]);
    }
    activeSession = session;
    session.updateFooter();

    if (entriesList) {
      clearChildren(entriesList);
      entriesList.removeAttribute("aria-busy");
    }
    if (folderBanner) {
      folderBanner.hidden = true;
    }

    dialog.returnValue = "";
    if (!dialog.open) {
      dialog.showModal();
    }
    session.browse(cfg.initialPath, { allowFallback: Boolean(cfg.initialPath) });
  });
}
