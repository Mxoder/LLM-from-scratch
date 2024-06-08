用 PyTorch 从零实现 LoRA。

在本文你将看到：

- 如何用 pytorch 从零实现 LoRA，表现会近似于 huggingface 的 peft 库（代码实现不一定）
- 如何卸载保存自实现的 LoRA 适配器和重载自实现的 LoRA 适配器
- 如何用自实现的 LoRA 微调 LLM，支持 hugging face 格式的模型
- 如何合并自实现的 LoRA 适配器，并上传到 hugging face
- （待更新）不同的 LoRA 初始化方法
- （待更新）不同的 LoRA 改进方法

在本文看不到：

- LoRA 的应用原理详解
- LoRA 的优化原理详解
- QLoRA、INT8 等量化 LoRA 方式