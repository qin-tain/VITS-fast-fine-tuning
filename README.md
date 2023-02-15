English Documentation Please Click [here](https://github.com/SongtingLiu/VITS_voice_conversion/blob/main/README_EN.md)
# VITS 声线转换
这个代码库会指导你如何将自己的声线通过微调加入已有的VITS模型中，从而使得一个模型就可以实现用户声线到上百个角色声线的高质量转换。  

欢迎体验微调所使用的底模，一个包含中日英三语的TTS（文本到语音合成）模型！ 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer)

### 目前支持的任务:
- [x] 转换用户声线到 [这些角色](https://github.com/SongtingLiu/VITS_voice_conversion/blob/main/configs/finetune_speaker.json)
- [ ] 自定义角色的中日英三语TTS（待完成）

### 目前支持声线转换和中日英三语TTS的角色
- [x] 赛马娘 （仅已实装角色）
- [x] 魔女的夜宴（柚子社） （5人）
- [x] 原神 （仅已实装角色）
- [ ] 任意角色（待完成）




## 微调
建议使用 [Google Colab](https://colab.research.google.com/drive/1omMhfYKrAAQ7a6zOCsyqpla-wU-QyfZn?usp=sharing)
进行微调任务，因为VITS在多语言情况下的某些环境依赖相当难以配置。
### 在Google Colab里，我需要花多长时间？
1. 安装依赖 (2 min)
2. 录入你自己的声音，至少20条3~4秒的短句 (5 min)
3. 进行微调 (30 min)  

微调结束后可以直接下载微调好的模型，日后在本地运行（不需要GPU）

## 本地运行和推理
0. 记得下载微调好的模型和config文件！
1. 下载最新的Release包（在Github页面的右侧）
2. 把下载的模型和config文件放在 `VC_inference`文件夹下, 确保模型的文件名为 `G_latest.pth` ，config文件名为 `finetune_speaker.json`
3. 一切准备就绪后，文件结构应该如下所示:
```shell
VC_inference
├───VC_inference.exe
├───...
├───finetune_speaker.json
└───G_latest.json
```
4. 运行 `VC_inference.exe`, 浏览器会自动弹出窗口.

