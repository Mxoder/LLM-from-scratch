import json
import asyncio
from tqdm import tqdm
from loguru import logger
from collections import deque
from openai import AsyncOpenAI

# 初始化 OpenAI 客户端
logger.info("Initializing OpenAI client...")
client = AsyncOpenAI(
    api_key="your-api-key",  # 替换为你的 api_key
    # base_url="your-base-url",  # 替换为你的 base_url，若调用 openai 官方则不用填
)
logger.info("OpenAI client initialized.")

# 配置文件路径
data_file = "stories.jsonl"

# 公用缓冲区
buffer = deque()


# 写入文件，追加写入
def write_to_file(data, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def get_response_with_retry(prompt, semaphore, model, max_retries=5):
    for attempt in range(max_retries):
        async with semaphore:
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                    max_tokens=512,
                )
                content = completion.choices[0].message.content
                return content
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get response: {str(e)}")
                    return None
                await asyncio.sleep(2**attempt)

    logger.error(
        f"Failed to get response after {max_retries} retries for prompt '{prompt}'."
    )
    return None


async def main(semaphore_num: int, data_num: int, buffer_size: int):
    """
    主函数：异步处理多个任务。

    参数：
        semaphore_num (int): 并发信号量的数量，控制同时运行的任务数。
        data_num (int): 设定的生成的故事条数。
        buffer_size (int): 缓冲区大小，达到该大小时将缓冲区内容写入文件。
    """
    semaphore = asyncio.Semaphore(semaphore_num)
    lock = asyncio.Lock()  # 用于同步写入缓冲区
    pbar = tqdm(total=data_num, desc="Processing")

    # 生成故事
    prompts = ["讲一个吸引人的、简短的故事。"] * data_num
    tasks = [
        get_response_with_retry(prompt, semaphore, model="Qwen/Qwen2-1.5B-Instruct")
        for prompt in prompts
    ]

    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            async with lock:  # 同步写入缓冲区
                buffer.append({"text": result})  # 将结果添加到缓冲区，形式任意
                pbar.update(1)  # 更新进度条
                if len(buffer) >= buffer_size:  # 如果缓冲区满了，写入文件
                    write_to_file(buffer, data_file)
                    buffer.clear()
        except Exception as e:
            logger.error(f"Error occurred while processing: {e}")
            continue

    # 处理剩余的缓冲区内容
    async with lock:
        if buffer:
            write_to_file(buffer, data_file)
            buffer.clear()

    pbar.close()
    logger.info("All tasks completed.")


if __name__ == "__main__":
    asyncio.run(main(semaphore_num=20, data_num=1000, buffer_size=100))
