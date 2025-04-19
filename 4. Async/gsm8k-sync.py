import json
import time
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

# 初始化 OpenAI 客户端
logger.info("Initializing OpenAI client...")
client = OpenAI(
    api_key="your-api-key",  # 替换为你的 api_key
    # base_url="your-base-url",  # 替换为你的 base_url，若调用 openai 官方则不用填
)
logger.info("OpenAI client initialized.")


def get_response_with_retry(prompt, model, max_retries=5):
    """
    同步函数：向 OpenAI 发送请求并处理重试逻辑。

    参数：
        prompt (str): 输入的问题或提示。
        model (str): 使用的模型名称。
        max_retries (int): 最大重试次数，默认为 5。

    返回：
        str: OpenAI 返回的响应内容；如果失败则返回 None。
    """
    for attempt in range(max_retries):  # 尝试最多 max_retries 次
        try:
            # 调用 OpenAI 的同步 API 获取响应
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},  # 构造用户输入的消息
                ],
                temperature=0.0,  # 设置温度参数为 0，确保结果更确定
                max_tokens=512,  # 限制最大生成 token 数
            )
            content = completion.choices[0].message.content  # 提取生成的内容
            return content
        except Exception as e:
            # 如果达到最大重试次数，记录错误并返回 None
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to get response after {max_retries} retries: {str(e)}"
                )
                return None
            # 指数退避策略：每次失败后等待时间加倍
            time.sleep(2**attempt)  # 指数退避策略

    # 如果所有尝试都失败，记录错误日志并返回 None
    logger.error(
        f"Failed to get response after {max_retries} retries for prompt '{prompt}'."
    )
    return None


def main():
    """
    主函数：同步处理多个任务。
    """
    # 读取测试数据
    with open("gsm8k-test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    start_time = time.time()  # 记录开始时间

    # 遍历数据，逐个处理问题
    for item in tqdm(data, desc="Processing"):  # 使用 tqdm 显示进度条
        prompt = item["question"]  # 提取问题文本
        response = get_response_with_retry(
            prompt,
            model="Qwen/Qwen2-1.5B-Instruct",  # 使用的模型名称
        )

    end_time = time.time()  # 记录结束时间
    logger.info(f"Elapsed time: {end_time - start_time:.2f} seconds")  # 打印总耗时


if __name__ == "__main__":
    main()
