import json
import time
import asyncio
from tqdm import tqdm
from loguru import logger
from openai import AsyncOpenAI

# 初始化 OpenAI 客户端
logger.info("Initializing OpenAI client...")
client = AsyncOpenAI(
    api_key="your-api-key",  # 替换为你的 api_key
    # base_url="your-base-url",  # 替换为你的 base_url，若调用 openai 官方则不用填
)
logger.info("OpenAI client initialized.")


async def get_response_with_retry(prompt, semaphore, model, max_retries=5):
    """
    异步函数：向 OpenAI 发送请求并处理重试逻辑。

    参数：
        prompt (str): 输入的问题或提示。
        semaphore (asyncio.Semaphore): 控制并发数的信号量。
        model (str): 使用的模型名称。
        max_retries (int): 最大重试次数，默认为 5。

    返回：
        str: OpenAI 返回的响应内容；如果失败则返回 None。
    """
    for attempt in range(max_retries):  # 尝试最多 max_retries 次
        async with semaphore:  # 使用信号量控制并发
            try:
                # 调用 OpenAI 的异步 API 获取响应
                completion = await client.chat.completions.create(
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
                await asyncio.sleep(2**attempt)

    # 如果所有尝试都失败，记录日志并返回 None
    logger.error(
        f"Failed to get response after {max_retries} retries for prompt '{prompt}'."
    )
    return None


async def main(semaphore_num: int):
    """
    主函数：异步处理多个任务。

    参数：
        semaphore_num (int): 并发信号量的数量，控制同时运行的任务数。
    """
    # 创建一个信号量对象，限制并发任务数
    semaphore = asyncio.Semaphore(semaphore_num)

    # 读取测试数据
    with open("gsm8k-test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化进度条
    pbar = tqdm(total=len(data), desc="Processing")

    # 创建异步任务列表，每个任务调用 get_response_with_retry 处理一个问题
    tasks = [
        get_response_with_retry(
            item["question"],  # 问题文本
            semaphore,  # 并发控制信号量
            model="Qwen/Qwen2-1.5B-Instruct",  # 使用的模型名称
        )
        for item in data
    ]

    start_time = time.time()  # 记录开始时间

    # 使用 asyncio.as_completed 处理任务，按完成顺序更新进度条
    for future in asyncio.as_completed(tasks):
        try:
            result = await future  # 等待任务完成
            pbar.update(1)  # 更新进度条
        except Exception as e:
            # 捕获异常并记录错误日志
            logger.error(f"Error occurred while processing: {e}")
            continue

    pbar.close()  # 关闭进度条
    end_time = time.time()  # 记录总耗时
    logger.info(f"Elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main(semaphore_num=40))  # 启动主函数，设置并发信号量数量为 40
