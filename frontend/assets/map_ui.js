/**
 * Spatial Quest — Interactive Facility Blueprint Map, Markdown Formatter & Voice Helper
 */
(function (global) {
  "use strict";

  const ROOM_BOXES = [
    { id: "reception", label: "1. Reception Hub", x1: -12, x2: 12, z1: -20.5, z2: -10, fill: "#131c2b" },
    { id: "library", label: "3. Library / Archives", x1: -12, x2: -2.2, z1: -10, z2: 0, fill: "#171d29" },
    { id: "corridor", label: "Corridor", x1: -2.2, x2: 2.2, z1: -10, z2: 0, fill: "#0e1522" },
    { id: "office", label: "2. Research Office", x1: 2.2, x2: 12, z1: -10, z2: 0, fill: "#151f2e" },
    { id: "laboratory", label: "4. Investigation Lab", x1: -12, x2: 12, z1: 0, z2: 8, fill: "#142031" },
    { id: "storage", label: "5. Storage Vault", x1: -12, x2: 12, z1: 8, z2: 22.3, fill: "#111722" },
  ];

  class FacilityMapRenderer {
    constructor(canvas, onWaypointClick) {
      this.canvas = canvas;
      this.ctx = canvas ? canvas.getContext("2d") : null;
      this.onWaypointClick = onWaypointClick;

      if (this.canvas) {
        this.canvas.addEventListener("click", (e) => {
          const rect = this.canvas.getBoundingClientRect();
          const px = ((e.clientX - rect.left) / rect.width) * this.canvas.width;
          const py = ((e.clientY - rect.top) / rect.height) * this.canvas.height;
          const world = this.canvasToWorld(px, py);
          if (this.onWaypointClick) this.onWaypointClick(world.x, world.z);
        });
      }
    }

    worldToCanvas(wx, wz) {
      // Facility world X: [-13.5, 13.5], Z: [-22, 23.5]
      // Map rotated cleanly so North (+Z) is top, East (+X) is right!
      const w = this.canvas.width;
      const h = this.canvas.height;
      const pad = 28;
      const u = ((wx + 13.5) / 27.0) * (w - pad * 2) + pad;
      const v = h - pad - ((wz + 21.5) / 45.0) * (h - pad * 2);
      return { x: u, y: v };
    }

    canvasToWorld(cx, cy) {
      const w = this.canvas.width;
      const h = this.canvas.height;
      const pad = 28;
      const wx = ((cx - pad) / (w - pad * 2)) * 27.0 - 13.5;
      const wz = ((h - pad - cy) / (h - pad * 2)) * 45.0 - 21.5;
      return {
        x: Math.max(-11.4, Math.min(11.4, wx)),
        z: Math.max(-19.2, Math.min(21.0, wz)),
      };
    }

    render(state) {
      if (!this.ctx) return;
      const ctx = this.ctx;
      const w = this.canvas.width;
      const h = this.canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#080c14";
      ctx.fillRect(0, 0, w, h);

      // Draw subtle architectural grid
      ctx.strokeStyle = "rgba(148, 163, 184, 0.06)";
      ctx.lineWidth = 1;
      for (let gx = -12; gx <= 12; gx += 4) {
        const a = this.worldToCanvas(gx, -21);
        const b = this.worldToCanvas(gx, 22.5);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }

      // Draw room zones
      ROOM_BOXES.forEach((rm) => {
        const p1 = this.worldToCanvas(rm.x1, rm.z2);
        const p2 = this.worldToCanvas(rm.x2, rm.z1);
        ctx.fillStyle = rm.fill;
        ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

        ctx.fillStyle = "rgba(148, 163, 184, 0.72)";
        ctx.font = "600 11px Inter, sans-serif";
        ctx.fillText(rm.label, p1.x + 8, p1.y + 16);
      });

      // Door 3 Gate Indicator
      const d3a = this.worldToCanvas(-1.8, 8.0);
      const d3b = this.worldToCanvas(1.8, 8.0);
      ctx.strokeStyle = state.door3Unlocked ? "#10b981" : "#f43f5e";
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(d3a.x, d3a.y); ctx.lineTo(d3b.x, d3b.y); ctx.stroke();

      // Active navigation route waypoints if any
      if (state.waypoints && state.waypoints.length > 0) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        const startP = this.worldToCanvas(state.playerPos.x, state.playerPos.z);
        ctx.moveTo(startP.x, startP.y);
        state.waypoints.forEach((wp) => {
          const p = this.worldToCanvas(wp.x, wp.z);
          ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Interactable objects
      (state.interactables || []).forEach((obj) => {
        const p = this.worldToCanvas(obj.position[0], obj.position[2]);
        const isSel = state.selectedId === obj.id;
        ctx.fillStyle = isSel ? "#f59e0b" : "#94a3b8";
        ctx.beginPath();
        ctx.arc(p.x, p.y, isSel ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fill();
        if (isSel) {
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
          ctx.stroke();
        }
      });

      // Investigator arrow & FOV cone
      const ip = this.worldToCanvas(state.playerPos.x, state.playerPos.z);
      const fwdX = -Math.sin(state.yaw);
      const fwdZ = -Math.cos(state.yaw);
      // Convert forward vector into canvas angle (Y axis inverted)
      const angle = Math.atan2(-fwdZ, fwdX);

      ctx.save();
      ctx.translate(ip.x, ip.y);
      ctx.rotate(angle);

      // Soft vision wedge
      ctx.fillStyle = "rgba(56, 189, 248, 0.16)";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, 38, -0.45, 0.45);
      ctx.closePath();
      ctx.fill();

      // Player triangle pointer
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(9, 0);
      ctx.lineTo(-6, -6);
      ctx.lineTo(-3, 0);
      ctx.lineTo(-6, 6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  function escapeHtml(raw) {
    return String(raw || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatInlineMarkdown(escapedLine) {
    let html = escapedLine;
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return html;
  }

  function renderMarkdownLite(rawText) {
    let normalized = String(rawText || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    // Repair accidental line wraps inside numbers/codes without merging numbered list items
    normalized = normalized.replace(/(\d)\n+(\d+(?!\s*[.)](?:\s|$)))/g, "$1$2");
    normalized = normalized.replace(/(\d\.)\n+(\d+(?!\s*[.)](?:\s|$)))/g, "$1$2");

    const lines = normalized.split("\n");
    const blocks = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();
      if (!trimmed) {
        blocks.push('<div class="chat-spacer"></div>');
        continue;
      }
      const hMatch = trimmed.match(/^#{1,4}\s+(.*)$/);
      if (hMatch) {
        blocks.push(`<div class="chat-heading">${formatInlineMarkdown(escapeHtml(hMatch[1]))}</div>`);
        continue;
      }
      const bulletMatch = trimmed.match(/^(?:[-*•])\s+(.*)$/);
      if (bulletMatch) {
        blocks.push(`<div class="chat-list-item"><span class="chat-bullet">•</span><span>${formatInlineMarkdown(escapeHtml(bulletMatch[1]))}</span></div>`);
        continue;
      }
      blocks.push(`<div class="chat-line">${formatInlineMarkdown(escapeHtml(line))}</div>`);
    }
    return blocks.join("");
  }

  class VoiceController {
    constructor(onTranscript, onStateChange) {
      this.onTranscript = onTranscript;
      this.onStateChange = onStateChange;
      this.ttsEnabled = false;
      this.isListening = false;
      const SpeechRec = global.SpeechRecognition || global.webkitSpeechRecognition;
      if (SpeechRec) {
        this.recognition = new SpeechRec();
        this.recognition.lang = "en-US";
        this.recognition.interimResults = false;
        this.recognition.onresult = (e) => {
          const text = e.results[0] && e.results[0][0] ? e.results[0][0].transcript : "";
          if (text && this.onTranscript) this.onTranscript(text);
        };
        this.recognition.onend = () => {
          this.isListening = false;
          if (this.onStateChange) this.onStateChange(this.isListening, this.ttsEnabled);
        };
      } else {
        this.recognition = null;
      }
    }

    toggleListening() {
      if (!this.recognition) return false;
      if (this.isListening) {
        this.recognition.stop();
        this.isListening = false;
      } else {
        this.isListening = true;
        this.recognition.start();
      }
      if (this.onStateChange) this.onStateChange(this.isListening, this.ttsEnabled);
      return true;
    }

    toggleTTS() {
      this.ttsEnabled = !this.ttsEnabled;
      if (!this.ttsEnabled && global.speechSynthesis) {
        global.speechSynthesis.cancel();
      }
      if (this.onStateChange) this.onStateChange(this.isListening, this.ttsEnabled);
      return this.ttsEnabled;
    }

    speak(text) {
      if (!this.ttsEnabled || !global.speechSynthesis) return;
      global.speechSynthesis.cancel();
      const cleaned = String(text || "").replace(/[*`#•🎯🗺️👁️📚🧭ℹ️]/g, "");
      const utter = new SpeechSynthesisUtterance(cleaned.slice(0, 320));
      utter.rate = 1.04;
      global.speechSynthesis.speak(utter);
    }
  }

  global.SpatialUIHelpers = {
    FacilityMapRenderer,
    renderMarkdownLite,
    VoiceController,
  };
})(window);

