/**
 * web/static/js/fs_picker.js
 * Safe Local Disk & Folder Browser Modal (<dialog closedby="any">):
 * - Supports multi-file + whole-folder selection for Queue, or single-file selection for Compare/Subtrans.
 * - Displays natural episode order, shows.yaml routing badges, drive free space, and already-upscaled hints.
 */

import { apiFetch, clearChildren, el, initDialogLightDismiss, showToast } from "./ui.js";

let initialized = false;
let activeSession = null;
let lastVisitedPath = "";

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

  const cfg = {
    title: options.title || "Chọn Video hoặc Thư mục trên Máy trạm",
    multi: Boolean(options.multi),
    allowFolders: Boolean(options.allowFolders),
    includeSubtitles: Boolean(options.includeSubtitles),
    allow4k: Boolean(options.allow4k),
    initialPath: options.initialPath ?? (lastVisitedPath || ""),
  };

  if (titleEl) {
    titleEl.textContent = cfg.title;
  }

  return new Promise((resolve) => {
    const session = {
      selected: new Set(),
      currentPath: "",
      parentPath: null,
      workspaceRoot: "",
      browseSeq: 0,
      resolved: false,
      resolve,
      updateFooter() {
        const count = this.selected.size;
        if (selectionCount) {
          selectionCount.textContent =
            count > 0
              ? `Đã chọn ${count} mục.`
              : cfg.multi
                ? "Nhấp vào thư mục để mở, hoặc đánh dấu các tệp cần thêm."
                : "Nhấp vào một tệp để chọn nhanh.";
        }
        if (btnConfirm instanceof HTMLButtonElement) {
          btnConfirm.disabled = count === 0;
        }
      },
      async browse(targetPath) {
        if (!entriesList || activeSession !== this || this.resolved) return;
        const reqId = ++this.browseSeq;
        const isCurrent = () =>
          activeSession === this && !this.resolved && dialog.open && reqId === this.browseSeq;

        clearChildren(entriesList);
        entriesList.append(el("p", { className: "empty-hint", textContent: "Đang đọc thư mục…" }));

        try {
          const qs = new URLSearchParams();
          if (targetPath) qs.set("path", targetPath);
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
          if (btnUp instanceof HTMLButtonElement) {
            btnUp.disabled = !this.currentPath;
          }

          // Folder whole-select banner
          if (folderBanner && folderSummary) {
            if (cfg.allowFolders && this.currentPath && Number(data.source_video_count || 0) > 0) {
              const routeModel = data.folder_routing?.model_id || "animevideov3";
              const routePol = data.folder_routing?.dedup_policy || "default";
              folderSummary.textContent = `Thư mục này có ${data.source_video_count} tập nguồn • Định tuyến: ${routeModel} (${routePol})`;
              folderBanner.hidden = false;
            } else {
              folderBanner.hidden = true;
            }
          }

          clearChildren(entriesList);
          const entries = Array.isArray(data.entries) ? data.entries : [];
          let renderedCount = 0;
          for (const item of entries) {
            if (item.type === "video_4k" && !cfg.allow4k && !cfg.multi) {
              // Still show in queue browser as informational, hide only when picking 1080p source
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
                textContent: "Thư mục này chưa có thư mục con hoặc tệp media phù hợp.",
              })
            );
          }
          this.updateFooter();
        } catch (err) {
          if (!isCurrent()) return;
          clearChildren(entriesList);
          entriesList.append(
            el("p", { className: "empty-hint", textContent: `Lỗi đọc thư mục: ${err.message}` })
          );
          showToast(err.message, "error");
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
              onClick: () => this.browse(item.path),
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
              onClick: () => {
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
              onClick: () => {
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

    dialog.returnValue = "";
    if (!dialog.open) {
      dialog.showModal();
    }
    session.browse(cfg.initialPath);
  });
}
