import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gradio as gr


@dataclass
class DemoResult:
    tokens: List[str]
    entities: List[Dict[str, Any]]
    notes: str


def _simple_tokenize(text: str) -> List[str]:
    # 轻量级可视化：不依赖模型，仅用于前端展示
    text = (text or "").strip()
    if not text:
        return []
    return [t for t in text.replace("\n", " ").split(" ") if t]


def _dummy_infer(text: str, task: str, do_load: bool, local_path: str) -> Tuple[str, str, str]:
    tokens = _simple_tokenize(text)

    # 伪造的“实体识别/答案抽取”结果，用于可视化
    entities: List[Dict[str, Any]] = []
    if tokens:
        if task == "命名实体识别（NER）":
            # 仅示例：把第一个 token 当作“疾病/基因”等
            entities.append({"span": tokens[0], "label": "BIO-ENTITY", "score": 0.99})
        else:
            entities.append({"answer": tokens[0], "confidence": 0.88})

    notes = (
        "本 WebUI 的目标是提供可视化与流程演示。为了避免在模板任务中下载大体积权重文件，"
        "此处不进行真实推理；若后续需要接入真实模型，可在本地放置权重并开启‘尝试加载本地模型’。"
    )

    if do_load:
        # 这里故意不真正加载；只提示用户如何接入
        if local_path.strip():
            notes += f"\n\n已收到本地模型路径：{local_path.strip()}（当前为演示模式，不会实际加载）。"
        else:
            notes += "\n\n你开启了加载开关，但未提供本地模型路径。"

    tokens_json = json.dumps({"tokens": tokens}, ensure_ascii=False, indent=2)
    entities_json = json.dumps({"task": task, "result": entities}, ensure_ascii=False, indent=2)

    # 简单“高亮”预览：用方括号标注第一个 token
    preview = text
    if tokens:
        preview = text.replace(tokens[0], f"【{tokens[0]}】", 1)

    return preview, tokens_json, entities_json


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="BioBERT Base Cased v1.1 - WebUI（演示）") as demo:
        gr.Markdown(
            """
<div class="hero">
  <h2>BioBERT Base Cased v1.1：可视化演示 WebUI</h2>
  <p>本界面面向“模型加载—输入构造—结果可视化”的最小闭环展示，强调<strong>可复现的交互路径</strong>与<strong>可解释的输出呈现</strong>。为了控制模板体积与下载成本，默认以演示推理替代真实权重推理。</p>
</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                text = gr.Textbox(
                    label="输入文本（可为医学摘要/病历片段/论文句子）",
                    placeholder="例如：EGFR mutation was detected in lung cancer patients.",
                    lines=6,
                )
                task = gr.Dropdown(
                    ["命名实体识别（NER）", "抽取式问答（QA）"],
                    value="命名实体识别（NER）",
                    label="任务类型",
                )
                with gr.Row():
                    do_load = gr.Checkbox(label="尝试加载本地模型（演示模式，不会真正加载）", value=False)
                    local_path = gr.Textbox(label="本地模型路径（可选）", placeholder="例如：./hf_repo 或 ./weights", scale=2)

                run = gr.Button("运行（演示推理）", variant="primary")

            with gr.Column(scale=5):
                preview = gr.Textbox(label="可视化预览（高亮示例）", lines=6)

        with gr.Row():
            tokens_json = gr.Code(label="Token 序列（JSON）", language="json")
            entities_json = gr.Code(label="结构化结果（JSON）", language="json")

        run.click(
            _dummy_infer,
            inputs=[text, task, do_load, local_path],
            outputs=[preview, tokens_json, entities_json],
        )

        gr.Markdown(
            """
**说明**：本项目在论文式写作中更强调“方法—实现—可视化”的逻辑一致性，因此界面不追求复杂任务覆盖，而以关键链路的可见性与可扩展性为核心。
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
        .container {max-width: 1100px !important;}
        .hero {border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 18px; background: linear-gradient(135deg, rgba(137,86,255,.08), rgba(255,210,30,.10));}
        .mono textarea, .mono pre {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace !important;}
        """,
    )
