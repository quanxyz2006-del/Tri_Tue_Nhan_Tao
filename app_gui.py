"""Thread-safe CustomTkinter front end for the English–Vietnamese trainer."""

import copy
import json
import math
from pathlib import Path
import queue
import threading
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent
PARAMETERS = (
    ("embedding_size", "Embedding size", "128"),
    ("hidden_size", "Hidden units", "256"),
    ("learning_rate", "Learning rate", "0.001"),
    ("epochs", "Epochs", "20"),
    ("batch_size", "Batch size", "32"),
    ("max_length", "Độ dài tối đa", "50"),
    ("seed", "Random seed", "42"),
)


def run_worker(events, stop_event, operation, config, path, text, trainer):
    """Use only ordinary Python objects here; never call Tk from this thread."""
    def callback(*args, **kwargs):
        events.put(("progress", copy.deepcopy((args, kwargs))))

    try:
        if stop_event.is_set():
            events.put(("cancelled", None))
            return
        if operation in ("train", "ablation", "load"):
            # Import lazily so dependency/import errors appear in the GUI.
            from src.trainer import Trainer

            candidate = Trainer(config)
        else:
            candidate = trainer
        if operation == "load":
            candidate.load(path)
            result = None
        elif operation in ("train", "ablation"):
            candidate.prepare(path)
            if stop_event.is_set():
                events.put(("cancelled", None))
                return
            if operation == "train":
                result = candidate.fit(
                    bidirectional=True, callback=callback, stop_event=stop_event
                )
            else:
                result = candidate.ablation(callback=callback, stop_event=stop_event)
        elif operation == "translate":
            result = candidate.translate(text)
        elif operation == "save":
            candidate.save(path)
            result = None
        else:
            raise ValueError(f"Unknown operation: {operation}")
        if stop_event.is_set():
            events.put(("cancelled", None))
        else:
            events.put(("success", (operation, candidate, result)))
    except Exception as exc:
        if stop_event.is_set():
            events.put(("cancelled", None))
        else:
            events.put(("error", (str(exc), traceback.format_exc())))


class TranslationApp(ctk.CTk):
    """All widget access and plotting happen in the main event loop."""

    def __init__(self, autoload=True):
        super().__init__()
        self.title("Dịch máy Anh – Việt | biLSTM Encoder–Decoder")
        self.geometry("1200x850")
        self.minsize(960, 700)
        self.trainer = None
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.busy = False
        self.closing = False
        self.results = {}
        self.live_results = {}
        self.controls = []
        self.entries = {}
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_controls()
        self._build_output()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.poll_id = self.after(100, self._poll)
        self._set_busy(False)
        checkpoint = ROOT / "artifacts" / "basic_15min" / "bilstm.pt"
        if autoload and checkpoint.is_file():
            self._start("load", str(checkpoint))

    def _button(self, parent, text, command):
        button = ctk.CTkButton(parent, text=text, command=command)
        button.pack(fill="x", padx=12, pady=5)
        self.controls.append(button)
        return button

    def _build_controls(self):
        panel = ctk.CTkScrollableFrame(self, width=245)
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(panel, text="DỮ LIỆU & THAM SỐ", font=("Arial", 17, "bold")).pack(pady=10)
        self.data_path = ctk.CTkEntry(panel)
        self.data_path.insert(0, str(ROOT / "data" / "eng-vie.txt"))
        self.data_path.pack(fill="x", padx=12, pady=5)
        self.controls.append(self.data_path)
        self._button(panel, "Chọn dữ liệu Anh – Việt", self._choose_data)
        for key, label, default in PARAMETERS:
            ctk.CTkLabel(panel, text=label, anchor="w").pack(fill="x", padx=12)
            entry = ctk.CTkEntry(panel)
            entry.insert(0, default)
            entry.pack(fill="x", padx=12, pady=(0, 4))
            self.entries[key] = entry
            self.controls.append(entry)
        self._button(panel, "Huấn luyện biLSTM", lambda: self._start("train"))
        self._button(panel, "Chạy Ablation Study", lambda: self._start("ablation"))
        self._button(panel, "Nạp checkpoint", self._load)
        self.save_button = self._button(panel, "Lưu checkpoint", self._save)
        self.cancel_button = ctk.CTkButton(panel, text="Hủy tác vụ", command=self._cancel,
                                            fg_color="#a33b38", hover_color="#842e2b")
        self.cancel_button.pack(fill="x", padx=12, pady=8)

    def _build_output(self):
        panel = ctk.CTkFrame(self)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(6, weight=1)
        ctk.CTkLabel(panel, text="Văn bản tiếng Anh", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        self.source = ctk.CTkTextbox(panel, height=85)
        self.source.grid(row=1, column=0, sticky="ew", padx=12, pady=5)
        self.source.insert("1.0", "He sat on the bank of the river.")
        self.controls.append(self.source)
        self.translate_button = ctk.CTkButton(
            panel, text="Dịch Anh → Việt", command=lambda: self._start("translate"))
        self.translate_button.grid(row=2, column=0, sticky="w", padx=12, pady=5)
        self.controls.append(self.translate_button)
        ctk.CTkLabel(panel, text="Bản dịch tiếng Việt", anchor="w").grid(
            row=3, column=0, sticky="ew", padx=12)
        self.translation = ctk.CTkTextbox(panel, height=85, state="disabled")
        self.translation.grid(row=4, column=0, sticky="ew", padx=12, pady=5)
        self.status = ctk.CTkLabel(panel, text="Sẵn sàng — hãy huấn luyện hoặc nạp checkpoint.", anchor="w")
        self.status.grid(row=5, column=0, sticky="ew", padx=12, pady=5)
        self.tabs = ctk.CTkTabview(panel)
        self.tabs.grid(row=6, column=0, sticky="nsew", padx=8, pady=8)
        for name in ("Biểu đồ", "Các bước dịch", "Nhật ký"):
            tab = self.tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
        self.details = ctk.CTkTextbox(self.tabs.tab("Các bước dịch"), state="disabled")
        self.details.grid(row=0, column=0, sticky="nsew")
        self.logs = ctk.CTkTextbox(self.tabs.tab("Nhật ký"), state="disabled")
        self.logs.grid(row=0, column=0, sticky="nsew")
        self.figure = Figure(figsize=(8, 4), dpi=100, constrained_layout=True)
        self.loss_axis, self.bleu_axis = self.figure.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tabs.tab("Biểu đồ"))
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._plot()

    @staticmethod
    def _write(widget, text, append=False):
        widget.configure(state="normal")
        if not append:
            widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _log(self, text):
        self._write(self.logs, text + "\n", append=True)

    def _choose_data(self):
        path = filedialog.askopenfilename(parent=self, title="Chọn dữ liệu song ngữ",
                                          filetypes=[("Text / TSV", "*.txt *.tsv"), ("All files", "*.*")])
        if path:
            self.data_path.delete(0, "end")
            self.data_path.insert(0, path)

    def _load(self):
        path = filedialog.askopenfilename(parent=self, title="Nạp checkpoint",
                                          filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self._start("load", path)

    def _save(self):
        path = filedialog.asksaveasfilename(parent=self, title="Lưu checkpoint",
                                           defaultextension=".pt", filetypes=[("PyTorch checkpoint", "*.pt")])
        if path:
            self._start("save", path)

    def _config(self):
        config = {}
        for key, label, _ in PARAMETERS:
            try:
                raw = self.entries[key].get().strip()
                value = float(raw) if key == "learning_rate" else int(raw)
                minimum = 0 if key == "seed" else 1
                if key == "learning_rate":
                    valid = math.isfinite(value) and value > 0
                else:
                    valid = value >= minimum
                if not valid:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{label}: nhập số hợp lệ "
                                 "(seed ≥ 0; các tham số khác > 0).") from None
            config[key] = value
        if config["max_length"] < 2:
            raise ValueError("Độ dài tối đa phải ít nhất là 2.")
        return config

    def _start(self, operation, path=None):
        if self.busy or self.closing:
            return
        try:
            # Loading restores its own configuration from the checkpoint.
            config = self._config() if operation in ("train", "ablation") else {}
            text = self.source.get("1.0", "end").strip()
            if operation in ("translate", "save") and self.trainer is None:
                raise ValueError("Hãy huấn luyện hoặc nạp checkpoint trước.")
            if operation == "translate" and not text:
                raise ValueError("Hãy nhập câu tiếng Anh.")
            if operation in ("train", "ablation"):
                path = self.data_path.get().strip()
            if operation in ("train", "ablation", "load"):
                resolved = Path(path).expanduser()
                if not resolved.is_absolute():
                    resolved = ROOT / resolved
                if not resolved.is_file():
                    raise ValueError(f"Không tìm thấy tệp: {resolved}")
                path = str(resolved)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Dữ liệu chưa hợp lệ", str(exc), parent=self)
            return
        self.stop_event = threading.Event()
        self.live_results = {}
        self._set_busy(True)
        self.status.configure(text=f"Đang chạy: {operation}…")
        self._log(f"Bắt đầu {operation}" + (f": {path}" if path else ""))
        self.worker = threading.Thread(target=run_worker, daemon=True,
            args=(self.events, self.stop_event, operation, config, path, text, self.trainer))
        self.worker.start()

    def _set_busy(self, busy):
        self.busy = busy
        for control in self.controls:
            control.configure(state="disabled" if busy else "normal")
        for control in (self.translate_button, self.save_button):
            control.configure(state="normal" if not busy and self.trainer is not None else "disabled")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def _cancel(self):
        if self.busy:
            self.stop_event.set()
            self.cancel_button.configure(state="disabled")
            self.status.configure(text="Đang chờ tác vụ dừng an toàn…")
            self._log("Đã yêu cầu hủy. Tác vụ I/O hoặc dịch đang chạy cần hoàn tất trước khi dừng.")

    def _poll(self):
        try:
            # Bound work per tick so a burst of backend messages cannot freeze Tk.
            for _ in range(100):
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    args, kwargs = payload
                    self._log(json.dumps({"args": args, **kwargs}, ensure_ascii=False, default=str))
                    if len(args) == 1 and isinstance(args[0], dict) and "epoch" in args[0]:
                        row = args[0]
                        name = row.get("model", "biLSTM")
                        metrics = self.live_results.setdefault(name, {"history": [], "bleu": {}})
                        metrics["history"].append(row)
                        self._plot(self.live_results)
                        if not self.stop_event.is_set():
                            self.status.configure(text=f"{name} — Epoch {row['epoch']} | "
                                f"Loss {row['train_loss']:.4f} / Val {row['val_loss']:.4f}")
                    continue
                try:
                    if kind == "success" and not self.stop_event.is_set():
                        self._success(*payload)
                        self.status.configure(text="Hoàn tất.")
                    elif kind == "cancelled" or self.stop_event.is_set():
                        self._plot()
                        self.status.configure(text="Đã hủy. Mô hình trước được giữ nguyên.")
                        self._log("Đã hủy; không nhận mô hình/kết quả mới. Tệp backend đã lưu có thể vẫn tồn tại.")
                    elif kind == "error":
                        self._plot()
                        message, details = payload
                        self._log(details)
                        self.status.configure(text="Tác vụ thất bại; xem Nhật ký.")
                        if not self.closing:
                            messagebox.showerror("Lỗi tác vụ", message, parent=self)
                except Exception:
                    self._log(traceback.format_exc())
                    self.status.configure(text="Không thể hiển thị kết quả; xem Nhật ký.")
                finally:
                    self._set_busy(False)
        finally:
            if self.closing and not self.busy:
                self.destroy()
            else:
                self.poll_id = self.after(100, self._poll)

    def _success(self, operation, candidate, result):
        if operation in ("train", "ablation"):
            results = {"biLSTM": result} if operation == "train" else result
            self._plot(results)
            for name, metrics in results.items():
                self._log(f"{name}: {metrics['seconds']:.2f}s | BLEU: {metrics['bleu']}")
                if metrics["history"]:
                    self._log(f"Epoch cuối: {metrics['history'][-1]}")
                self._log(json.dumps({key: value for key, value in metrics.items()
                                      if key not in ("history", "bleu")},
                                     ensure_ascii=False, indent=2, default=str))
            self.results = results
            self.trainer = candidate
        elif operation == "load":
            name = "biLSTM" if candidate.bidirectional else "LSTM"
            results = {name: candidate.result} if candidate.result else {}
            self._plot(results)
            self.results = results
            self.trainer = candidate
            for key, entry in self.entries.items():
                entry.configure(state="normal")
                entry.delete(0, "end")
                entry.insert(0, str(candidate.config[key]))
                entry.configure(state="disabled")
            self._write(self.translation, "")
            self._write(self.details, "")
            self._log("Đã nạp checkpoint. Có thể dịch ngay.")
            self._log(json.dumps(candidate.result, ensure_ascii=False, indent=2, default=str))
        elif operation == "translate":
            lines = [f"Tokens: {result['tokens']}", f"IDs: {result['ids']}",
                     "\nTensor shapes:", json.dumps(result["shapes"], ensure_ascii=False, indent=2, default=str),
                     "\nCác bước giải mã:"]
            if result.get("truncated"):
                lines.insert(0, "Chú ý: câu đầu vào đã được cắt theo độ dài tối đa của mô hình.")
            for index, step in enumerate(result["steps"], 1):
                lines.append(f"{index:02d}. {step['token']} | P = {step['probability']:.6f}")
            self._write(self.details, "\n".join(lines))
            self._write(self.translation, result["text"])
            self.tabs.set("Các bước dịch")
        elif operation == "save":
            self._log("Đã lưu checkpoint.")

    def _plot(self, results=None):
        results = self.results if results is None else results
        self.loss_axis.clear()
        self.bleu_axis.clear()
        self.loss_axis.set(title="Train / Validation Loss", xlabel="Epoch", ylabel="Loss")
        self.bleu_axis.set(title="BLEU comparison", ylabel="BLEU (backend scale)")
        labels = [f"BLEU-{i}" for i in range(1, 5)]
        self.bleu_axis.set_xticks(range(4), labels)
        count = max(1, len(results))
        width = 0.8 / count
        for index, (name, result) in enumerate(results.items()):
            history = result["history"]
            epochs = [row["epoch"] for row in history]
            self.loss_axis.plot(epochs, [row["train_loss"] for row in history], label=f"{name} train")
            self.loss_axis.plot(epochs, [row["val_loss"] for row in history], "--", label=f"{name} val")
            positions = [x + (index - (count - 1) / 2) * width for x in range(4)]
            if result.get("bleu"):
                self.bleu_axis.bar(positions, [result["bleu"][key] for key in labels], width, label=name)
        for axis in (self.loss_axis, self.bleu_axis):
            axis.grid(axis="y", alpha=0.25)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        self.canvas.draw_idle()

    def _close(self):
        if self.closing:
            return
        self.closing = True
        if self.busy:
            self._cancel()
            self.status.configure(text="Đang dừng tác vụ trước khi đóng cửa sổ…")
        else:
            self.after_cancel(self.poll_id)
            self.destroy()


def main():
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    app = TranslationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
