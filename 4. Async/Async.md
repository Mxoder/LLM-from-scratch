# 我的 api 调用太慢了！LLM api 的异步调用加速

[TOC]

在做 LLM 相关研究的时候，我们难免需要调用 api，例如如下场景：

- 调用模型（如 GPT）做评测/打分
- 调用模型合成数据

但传统单线程串行的方式一条一条调用也太慢了，我们该如何尽可能地加速调用呢？此时就要祭出「异步编程」这一工具了。只要服务端允许，异步调用可以轻松且丝滑地完成大量并发调用。

在本文你将看到：

- 以 OpenAI SDK 为例，如何用 python 的 `asyncio` 库实现 LLM api 的异步调用
- 上述两个实际场景的异步调用示例

在本文看不到：

- 同步/异步编程的原理详解
- 单线程/多线程/多进程的原理详解
- 串行/并行/并发的原理详解

本文用到的所有代码可以在 [Mxoder/LLM-from-scratch](https://github.com/Mxoder/LLM-from-scratch) 找到。

---

## 1. 异步编程的浅显理解

### 1.1 一个例子

「异步编程」这个词现在早已不鲜见了，但我们还是有必要简单理解一下它的概念。

与「异步编程」相对应的词是「同步编程」，也就是我们平时最常写的代码逻辑。举一个小学数学常见的做饭的例子，就是那种烧水切菜同时进行的题目：

> 一名厨师需要完成两项任务：烧开水和切菜。烧开水需要 5 分钟，切菜需要 3 分钟。

采用同步编程的思想，这名厨师的做法就是：先烧水，将锅起好以后就**在锅前干等 5 分钟，直到水烧开。**然后厨师再去切菜，花费 3 分钟。总计 8 分钟。

如果是异步编程的思想，那么这名厨师的做法就是：先烧水，将锅起好以后就**转头去切菜，花费 3 分钟。**然后厨师再等水烧开。由于切菜的时间包含在等水烧开之内，因此总计用时只有 5 分钟。

可以看到，同步编程的思想就是要依次完成提交的任务。对于厨师而言，他必须一条一条完成任务清单上的活动，中间**哪怕他空闲出来，也不能干别的事情。**而异步编程的思想就是，任务提交以后，如果当前有**非活动核心的阻塞等待**（对应这里就是烧水不需要厨师做事情），那么**活动核心**就可以转移到其他任务。

秉持**浅显理解**的初衷，我们不需要理解活动核心具体是谁，就知道它是干活儿的那个就行；我们也不需要理解异步编程中不同任务之间是如何转移的，只要知道异步编程的行为表现是这样的就可以。

### 1.2 异步编程适合什么任务？

先给结论：异步编程适合**高并发、I/O 密集型**的任务。

什么是高并发、I/O 密集型的任务？举个例子，假设我们现在要用浏览器打开 100 个页面并查看它们的内容，每个页面点开以后是需要一定加载时间的。那么我们有两个选择：

- 点开一个网页以后就等着它转圈圈加载，直到它加载好，查看它的内容，再去点开下一个新的页面。
- 点开一个网页以后不管它的加载，而是继续点击新的网页，这样我们可以不断打开很多新的网页，点击新页面的速度不受限于网页加载速度，而如果之前打开的页面加载好了，我们就回头查看它的内容，查看完就接着打开新网页。

在上面这个例子中，假设网页加载出内容不需要时间，那么两种做法其实用时是一样的（严格来说，第二种做法会有更多的切换开销，但我们忽略）。可是问题在于页面加载出内容实际上是需要时间的，也就是说，这个任务的主要耗时不在于我们自己、并非**计算密集型**，而是在于网页本身的加载耗时、即 **I/O 密集型**。

这两种选择其实就是对应着同步编程和异步编程。异步编程的特性让它特别适合**数量多、且时间开销不在于计算核心自身**的任务。规范来说，这就是**高并发、I/O 密集型**的任务。I/O 密集型任务的特点就是主要耗时在于等待外部资源（如网络请求、文件读写、数据库查询），CPU 在等待期间处于空闲状态。

### 1.3 一些辨析

我们这里关注异步编程和一些常见名词的辨析，因此这里就不讨论诸如「并行/并发」、「多线程/多进程」这些名词的辨析了。

> 异步编程一定并发吗？

并发是任务实现的一种形式，宏观上来说，并发近似于很多个任务同时进行。异步编程是完成任务的手段，尽管异步编程往往会以并发的形式来完成任务，但它们并不等价或是互为因果。

例如异步编程中只提交一个任务，或者提交的任务没有重叠，那么最终形式上也不算并发；而对于并发而言，实现路径也不只有异步一种，例如常见的多线程/多进程也可以实现并发。

> 异步编程和多线程是不是一样的？

异步编程和多线程都是实现并发的方式，因此从表现效果上来说，二者非常相似。但假如我们用 1.2 中打开网页的例子，异步编程就是一个人坐在电脑前来回切换查看 100 个网页，而多线程就相当于安排了 10 个人一起来点击查看。

也就是说，异步编程通常是单线程、单个活动核心，而多线程顾名思义是有多个线程一起工作、有多个活动核心。因此异步编程相比于多线程有切换开销小、占用低的优势。



## 2. LLM api 的异步调用——两个实际案例

由于目前各家 API 服务提供商都兼容 OpenAI SDK，诸如 vLLM、SGLang、LMDeploy 等框架也都提供 OpenAI 兼容的 serving 范式，通过 OpenAI SDK 调用 LLM 已经成为事实上的通用 LLM 调用方法。

因此我们接下来都以 OpenAI SDK 为例，编程语言为 python。

### 2.1 LLM 评测

让我们先来看一个最简单的例子：调用 LLM 完成一些 Benchmarks 的评测。我们以 GSM8K 的测试集为例，具体数据可以在 [这里](https://huggingface.co/datasets/openai/gsm8k/viewer/main/test) 获取。如果你想直接看最终的代码，可以在 [Mxoder/LLM-from-scratch](https://github.com/Mxoder/LLM-from-scratch) 找到。

我们先回想一下，最简单的串行调用 LLM 是如何实现的？

```python
from openai import OpenAI

client = OpenAI()	# 初始化一个客户端

def get_response(...):		# 实现单条数据的调用
    ...
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    content = completion.choices[0].message.content
    ...

def main(...):
    data = ...	# 读取数据
    for item in data:		# 循环调用
        response = get_response(item, ...)	# 完成一条数据

if __name__ == "__main__":
    main(...)
```

上面就是一个非常简单的循环调用逻辑，实现一个处理单条数据的函数 `get_response`，再在主函数 `main` 中遍历数据调用。那么我们该如何改造为异步调用呢？**再次注意：我们这里不会太涉及异步编程的原理，因此你可能看不到为什么要这么改的细节原因。**

首先，我们先 import 一下：

```python
import asyncio
from openai import AsyncOpenAI
```

`asyncio` 是 python 的异步库，而 `AsyncOpenAI` 是 `openai` 提供的专门用于异步调用的类。

接下来，我们给所有**涉及到异步调用**的函数加上 `async` 关键字，将其改造为协程函数，不涉及到 api 调用的一些辅助函数不用加。并且再给最终调用的主函数入口加一层事件循环的包装启动（这里是针对 `main`）。我们不用在意什么是协程和事件循环，只需要知道这么做可以让函数能够异步调用。

```python
async def get_response(...):		# 加上 async
    ...

async def main(...):
    ...

if __name__ == "__main__":
    # 原本是：main(...)
    asyncio.run(main(...))			# main 外面多加了一层 asyncio.run
```

然后我们换用 `openai`的异步调用 client，调用的时候加上一个 `await` 关键字：

```python
client = AsyncOpenAI()

async def get_response(...):
    ...
    completion = await client.chat.completions.create(	# await 关键字
        model=model,
        messages=messages,
        **kwargs
    )
    # 其他暂时不变
    content = completion.choices[0].message.content
    ...
```

然后我们把普通的循环替换掉，先创建异步任务列表，然后用 `asyncio.as_completed`  迭代。经过 `async` 包装的协程函数返回的是一个协程对象，类似于迭代器：

```python
async def main(...):
    data = ...	# 读取数据
    
    # 创建异步任务列表，每个任务调用 get_response 处理一个问题
    tasks = [get_response(...) for item in data]
    
    # 使用 asyncio.as_completed 处理任务，这里的顺序是实际的任务完成顺序，不是提交顺序
    for future in asyncio.as_completed(tasks):
        result = await future  # 完成一条数据
    ...
```

这么做以后，已经能够实现基本的异步并发调用了。但是这样有一个问题，并发量相当于 data 全部的数目，假如不做多个 client 和多个 key 的负载均衡，那么这对于服务端的调用压力非常大（成千上万的并发调用）。因此，我们还需要加上一个并发数目的控制，这里用到多线程很熟悉的工具——信号量（semaphore）：

```python
async def get_response(semaphore, ...):
    async with semaphore:	# 使用信号量控制并发
        ...			# 原有代码逻辑
    
async def main(...):
    ...
    # 创建一个信号量对象，限制并发任务数, semaphore_num 类型为数字
    semaphore = asyncio.Semaphore(semaphore_num)
    
    # 传入信号量
    tasks = [get_response(semaphore, ...) for item in data]
    ...
```

这样一来，我们就真正实现了异步调用的改造。当然，为了避免撞 Rate Limit 墙，我们还可以给 `get_response` 加上指数退避重试。如果想查看实时进度，我们也可以用 `tqdm` 加上一个进度条。

我利用 GSM8K 的测试数据集做了实际的时间测试，在不考虑撞 Rate Limit 墙的情况下，可以认为 semaphore 设置的并发量就是相较于原本单条顺序执行的提速倍数。例如 semaphore 设置为 8 时，单条顺序执行的时间为 24 min+，而异步调用可以做到 3min+。

如果你想看看实际的全部代码并自己实际测试一下，你可以在 [Mxoder/LLM-from-scratch](https://github.com/Mxoder/LLM-from-scratch) 找到代码，其中 `gsm8k-sync.py` 是原始的同步代码，`gsm8k-async.py` 是异步改造后的代码。这里的代码包含了指数退避重试、进度条等等功能，没有复杂的封装，可以轻松地修改为其他任务。

### 2.2 利用 LLM 合成数据

假如我们现在要使用 LLM 来合成数据（例如造指令微调的训练数据）。倘若上面 LLM 评测的数据量不大、还可以一次性跑完以后最后在统一写入文件，那么对于合成数据这个场景，数据量非常大，我们显然不能一直将数据存在内存中，而是要定期写入磁盘持久化存储。

因此，这一个案例相比于上面多解决了一个**异步调用保存写入文件**的问题。同样地，如果你想直接看最终的代码，可以在 [Mxoder/LLM-from-scratch](https://github.com/Mxoder/LLM-from-scratch) 找到。

假设我们已经和 2.1 中一样改造好了一份异步调用的代码。但这次我们的任务变成利用 LLM 造数据，不妨还是造故事数据集。我们希望每生成一定量的内容，就保存写入到文件当中。

因此，我们可以设立一个缓冲区 `buffer`，它的作用就是充当临时容器，存储调用的结果。当缓冲区的大小达到我们设定的阈值 `buffer_size` 以后，我们就将缓冲区中的内容追加写入文件，并清空缓冲区：

```python
buffer = deque()	# 初始化缓冲区

def write_to_file(...):		# 同步文件写入，也可以用 aiofiles 改造为 async
    ...

async def main(...):
    for future in asyncio.as_completed(tasks):
        result = await future
        
        # 将结果添加到缓冲区，内容形式任意，这里假设每条数据格式是 {"text": ...}
        buffer.append({"text": result})  
        
        if len(buffer) >= buffer_size:  # 如果缓冲区满了，写入文件
            write_to_file(buffer, data_file)
            buffer.clear()	# 清空缓冲区

	# 处理剩余的缓冲区内容
    if buffer:
        write_to_file(buffer, data_file)
        buffer.clear()
```

如果你熟悉多线程编程的话，就会发现这里的 `buffer` 被多个活动单位同时写入，会不会出现竞争冒险问题呢？理论上来说，`append` 是一个线程安全的原子操作，不过为了保险和严谨，我们还是加上一把锁来控制同步写入，写法和多线程非常相似：

```python
async def main(...):
    lock = asyncio.Lock()  # 用于同步写入缓冲区
    ...
    for future in asyncio.as_completed(tasks):
        result = await future
        async with lock:  # 上锁，同步写入缓冲区
            buffer.append({"text": result})  
            ...
    ...
```

那么到这里，我们就完成了异步调用写入文件，现在可以放心的造数据了！和 2.1 类似的，在 [Mxoder/LLM-from-scratch](https://github.com/Mxoder/LLM-from-scratch) 中也有提供一个生成数据的示例，见 `story.py`。当运行这个文件的时候，会异步并发生成许多条短故事，并按照一定间隔存储于设定的文件当中。经过实测，并发数为 20 的时候，可以在 40s+ 内生成 1000 条短故事。

## 3. 结语

这次简单分享了一下调用 LLM api 中的异步调用的写法。讲异步编程的文章很多，因此本文重点不在于讲解异步编程的原理，而是放了两个实际的改造案例。完整代码都在 repo 中可以找到，实现尽可能轻量、规范，便于阅读的同时也便于随手改造适应其他任务。

异步调用 LLM api 的用途还是挺广的，不仅能加速各种闭源模型的调用，哪怕是自部署模型，只要 serving 起来充当服务端，也可以用异步客户端调用的方式加快得到下游推理结果。当然，异步编程是通用的编程范式，不局限于 OpenAI SDK 和 python，这里只是考虑到通用性和易于上手，以此作为示例。