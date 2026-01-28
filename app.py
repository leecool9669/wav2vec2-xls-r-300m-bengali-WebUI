import json
import os
import socket
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # (n, c) -> mono
    return audio.mean(axis=1)


def _plot_waveform(audio: np.ndarray, sr: int) -> plt.Figure:
    audio = _to_mono(audio)
    t = np.arange(audio.shape[0], dtype=np.float32) / float(sr)
    fig = plt.figure(figsize=(9, 2.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(t, audio, linewidth=0.8)
    ax.set_title("波形（Waveform）")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("幅值")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _dummy_asr(
    audio_in: Optional[Tuple[int, np.ndarray]],
    do_load: bool,
    local_path: str,
    decode_mode: str,
) -> Tuple[str, str, Any]:
    """
    说明：
    - 本函数不下载、不加载任何大体积权重文件，仅返回“可视化占位结果”
    - 重点是演示 WebUI 的输入、结果组织、以及可视化输出结构
    """
    notes: str = (
        "本 WebUI 的目标是给出一个可复现的工程闭环：输入音频—特征/波形可视化—CTC 解码结果组织。"
        "为避免模板任务引入大体积权重文件，本实现默认不进行真实推理；界面保留了加载本地模型的入口，"
        "以便后续在不改 UI 的前提下接入 Transformers + Wav2Vec2ForCTC。"
    )

    if do_load:
        if local_path.strip():
            notes += f"\n\n已收到本地模型路径：{local_path.strip()}（当前为演示模式，不会实际加载）。"
        else:
            notes += "\n\n你开启了加载开关，但未提供本地模型路径。"

    # 返回一个“孟加拉语”风格的占位转写；并给出可视化 token / 置信度结构
    transcript = "আমি আজ একটি ছোট পরীক্ষামূলক বাক্য লিখছি (Demo transcription placeholder)."
    token_detail: Dict[str, Any] = {
        "decode_mode": decode_mode,
        "tokens": ["আমি", "আজ", "একটি", "ছোট", "পরীক্ষামূলক", "বাক্য"],
        "ctc": {
            "blank_id": 0,
            "logits_shape": "[T, V] (placeholder)",
            "notes": "真实推理将产生逐帧 logits，并经由 CTC best-path 或 beam search + 语言模型重打分。",
        },
        "confidence": {"mean": 0.86, "min": 0.41, "max": 0.98},
    }
    token_json = json.dumps(token_detail, ensure_ascii=False, indent=2)

    if not audio_in:
        return transcript, token_json, None

    sr, audio = audio_in
    fig = _plot_waveform(audio, sr)
    return transcript, token_json, fig


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="wav2vec2-xls-r-300m-bengali WebUI（演示）") as demo:
        gr.Markdown(
            """
<div class="hero">
  <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <div>
      <h2 style="margin:0;">wav2vec2-xls-r-300m-bengali：ASR 可视化演示 WebUI</h2>
      <p style="margin:6px 0 0 0; opacity:.9;">
        本界面聚焦“音频输入—波形可视化—CTC 解码结果结构化呈现”的最小闭环。默认不下载/不加载大权重，仅展示前端交互与输出形态。
      </p>
    </div>
  </div>
</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                audio = gr.Audio(
                    label="输入音频（上传或录音均可）",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                decode_mode = gr.Dropdown(
                    ["CTC Greedy（占位）", "CTC Beam + 5-gram LM（占位）"],
                    value="CTC Greedy（占位）",
                    label="解码策略（演示）",
                )
                with gr.Row():
                    do_load = gr.Checkbox(label="尝试加载本地模型（演示模式，不会真正加载）", value=False)
                    local_path = gr.Textbox(
                        label="本地模型路径（可选）",
                        placeholder="例如：./hf_repo 或 ./weights（不建议放大权重到本模板）",
                        scale=2,
                    )
                run = gr.Button("运行（演示转写）", variant="primary")

            with gr.Column(scale=6):
                transcript = gr.Textbox(label="转写结果（Transcript）", lines=4)
                token_json = gr.Code(label="解码细节（JSON）", language="json")
                waveform = gr.Plot(label="波形可视化（Waveform）")

        run.click(
            _dummy_asr,
            inputs=[audio, do_load, local_path, decode_mode],
            outputs=[transcript, token_json, waveform],
        )

        gr.Markdown(
            """
**提示**：若后续需要接入真实推理，通常需引入 `transformers` 与 `torch`，并在回调中使用 `AutoProcessor` 对音频重采样与特征化，再用 `Wav2Vec2ForCTC` 产生 logits，最后用 CTC 解码得到文本。
            """
        )

    return demo


if __name__ == "__main__":
    # 允许在某些环境下通过 PORT 指定端口；否则自动选择可用端口
    port_env = os.environ.get("PORT")
    if port_env:
        port = int(port_env)
    else:
        port = None
        for p in range(7860, 7871):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    port = p
                    break
                except OSError:
                    continue

    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        css="""
        .container {max-width: 1150px !important;}
        .hero {border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 18px; background: linear-gradient(135deg, rgba(41,98,255,.10), rgba(255,210,30,.10));}
        """,
    )
