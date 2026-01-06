#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber
)

from unitree_sdk2py.idl.std_msgs.msg.dds_._String_ import String_

from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


# ================= 配置 =================

MODEL_PATH = "/home/dwh/local_models/Qwen2.5-1.5B-Instruct"

# 防止一句话反复触发（冷却时间）
GREETING_COOLDOWN_SEC = 5.0


# ================= 大模型 =================

def init_llm():
    print(">>> [AI] 加载本地模型 ...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(">>> [AI] 模型加载完成")
    return tokenizer, model


def is_greeting(text: str, tokenizer, model) -> bool:
    """
    判断是否为打招呼
    """
    quick_words = ["你好", "您好", "hello", "hi", "嗨", "哈喽"]
    if any(w in text.lower() for w in quick_words):
        return True

    prompt = f"判断用户输入是否是打招呼，是输出YES，否输出NO。输入：{text}"
    messages = [
        {"role": "system", "content": "你是分类器，只输出YES或NO。"},
        {"role": "user", "content": prompt}
    ]

    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=3,
        temperature=0.1
    )

    resp = tokenizer.decode(out[0], skip_special_tokens=True).upper()
    return "YES" in resp


# ================= 动作 =================

def perform_greeting(audio_client: AudioClient, loco_client: LocoClient):
    print(">>> [🤖] 执行打招呼动作")

    # 绿灯
    audio_client.LedControl(0, 255, 0)

    # 挥手
    loco_client.WaveHand()

    # 语音
    audio_client.TtsMaker("你好，很高兴见到你。", 0)

    time.sleep(2.0)

    # 恢复蓝灯
    audio_client.LedControl(0, 0, 255)


# ================= 主程序 =================

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <networkInterface>")
        sys.exit(1)

    iface = sys.argv[1]

    # 1. 初始化 LLM
    tokenizer, model = init_llm()

    # 2. 初始化 DDS
    print(">>> 初始化 DDS ...")
    ChannelFactoryInitialize(0, iface)

    # 3. 初始化 ASR 订阅
    print(">>> 订阅 rt/audio_msg ...")
    asr_sub = ChannelSubscriber("rt/audio_msg", String_)
    asr_sub.Init()

    # 4. 初始化机器人客户端
    print(">>> 初始化机器人客户端 ...")

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()

    loco_client = LocoClient()
    loco_client.SetTimeout(10.0)
    loco_client.Init()

    # 启动提示
    audio_client.SetVolume(80)
    audio_client.TtsMaker("系统已启动，等待语音指令。", 0)
    audio_client.LedControl(0, 0, 255)

    print(">>> 系统运行中（Ctrl+C 退出）")

    last_greeting_time = 0.0

    try:
        while True:
            msg = String_("")

            if asr_sub.Read(msg):
                text = msg.data.strip()
                if not text:
                    continue

                print("[ASR]", text)

                now = time.time()
                if now - last_greeting_time < GREETING_COOLDOWN_SEC:
                    continue

                if is_greeting(text, tokenizer, model):
                    perform_greeting(audio_client, loco_client)
                    last_greeting_time = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n>>> 退出程序")

    finally:
        audio_client.LedControl(0, 0, 0)


if __name__ == "__main__":
    main()
