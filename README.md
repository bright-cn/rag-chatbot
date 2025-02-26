# 创建一个使用 GPT-4o 和 SERP 数据的 RAG 聊天机器人

[![Promo](https://github.com/luminati-io/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://www.bright.cn/) 

本指南将解释如何使用 Python、GPT-4o 以及 Bright Data 的 SERP API 来构建一个能够生成更精确且富含上下文信息的 AI 回答的 RAG 聊天机器人。

1. [简介](#简介)  
2. [什么是 RAG？](#什么是-rag)  
3. [为什么要使用 SERP 数据来为 AI 模型提供信息](#为什么要使用-serp-数据来为-ai-模型提供信息)  
4. [在 Python 中使用 GPT 模型和 SERP 数据进行 RAG：分步教程](#在-python-中使用-gpt-模型和-serp-数据进行-rag分步教程)  
   1. [步骤 1：初始化 Python 项目](#步骤-1初始化-python-项目)  
   2. [步骤 2：安装所需的库](#步骤-2安装所需的库)  
   3. [步骤 3：准备项目](#步骤-3准备项目)  
   4. [步骤 4：配置 SERP API](#步骤-4配置-serp-api)  
   5. [步骤 5：实现 SERP 数据抓取逻辑](#步骤-5实现-serp-数据抓取逻辑)  
   6. [步骤 6：从 SERP URL 中提取文本](#步骤-6从-serp-url-中提取文本)  
   7. [步骤 7：生成 RAG Prompt](#步骤-7生成-rag-prompt)  
   8. [步骤 8：执行 GPT 请求](#步骤-8执行-gpt-请求)  
   9. [步骤 9：创建应用程序的 UI](#步骤-9创建应用程序的-ui)  
   10. [步骤 10：整合所有部分](#步骤-10整合所有部分)  
   11. [步骤 11：测试应用程序](#步骤-11测试应用程序)  
5. [结论](#结论)

## 什么是 RAG？

RAG，全称 [Retrieval-Augmented Generation](https://blogs.nvidia.comhttps://brightdata.com/blog/what-is-retrieval-augmented-generation/)，是一种将信息检索与文本生成相结合的 AI 方法。在 RAG 工作流程中，应用程序首先会从外部来源（如文档、网页或数据库）检索相关数据。然后，它将这些数据传递给 AI 模型，以便生成更具上下文相关性的回复。

RAG 能够增强像 GPT 这样的大型语言模型（LLM）的功能，使其可以访问并引用超出其原始训练数据范围的最新信息。在需要精确且具有上下文特定信息的场景中，RAG 方法至关重要，因为它能够提高 AI 生成回复的质量和准确性。

## 为什么要使用 SERP 数据来为 AI 模型提供信息

GPT-4o 的知识截止日期是 [2023 年 10 月](https://computercity.com/artificial-intelligence/knowledge-cutoff-dates-llms)，这意味着它无法访问该时间之后发生的事件或信息。然而，得益于 [GPT-4o 模型](https://openai.com/index/hello-gpt-4o/)能够通过 Bing 搜索集成实时获取互联网数据，它可以提供更实时、更详细且更具上下文意义的回复。

## 在 Python 中使用 GPT 模型和 SERP 数据进行 RAG：分步教程

本教程将指导你如何使用 OpenAI 的 GPT 模型来构建一个 RAG 聊天机器人。基本思路是：先从 Google 上与搜索词相关的优质页面中获取文本，并将其作为 GPT 请求的上下文。

最大的难点在于如何获取 SERP 数据。大多数搜索引擎都具备高级的反爬虫措施，用以阻止机器人对其页面的自动访问。关于详细说明，请参考我们关于 [如何在 Python 中抓取 Google](https://www.bright.cn/blog/web-data/scraping-google-with-python) 的指南。

为了简化抓取流程，我们将使用 [Bright Data 的 SERP API](https://www.bright.cn/products/serp-api)。

通过此 SERP 抓取器，你可以使用简单的 HTTP 请求，从 Google、DuckDuckGo、Bing、Yandex、Baidu 以及其他搜索引擎中轻松获取 SERP。

之后，我们将使用[无头浏览器 (headless browser)](https://www.bright.cn/blog/web-data/best-headless-browsers)从返回的所有 URL 中提取文本数据，并将其作为 GPT 模型在 RAG 工作流中的上下文。如果你希望直接使用 AI 实时获取网络数据，可以参考我们关于 [使用 ChatGPT 进行网页抓取](https://www.bright.cn/blog/web-data/web-scraping-with-chatgpt) 的文章。

本指南的所有代码都已上传到一个 GitHub 仓库中：

```bash
git clone https://github.com/Tonel/rag_gpt_serp_scraping
```

按照 README.md 文件中的指示来安装项目依赖并启动该项目。

请注意，本指南展示的方法可以轻松适配到其他搜索引擎或 LLM。

> **注意**：  
> 本教程适用于 Unix 和 macOS 环境。如果你使用 Windows，可以通过 [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) 进行类似操作。

### 步骤 #1：初始化 Python 项目

确保你的机器上安装了 Python 3。如果没有，请从 [Python 官网](https://www.python.org/downloads/) 下载并安装。

创建项目文件夹并在终端中切换到该文件夹：

```bash
mkdir rag_gpt_serp_scraping

cd rag_gpt_serp_scraping
```

`rag_gpt_serp_scraping` 文件夹将包含你的 Python RAG 项目。

然后，用你喜欢的 Python IDE（如 [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) 或 [安装了 Python 插件的 Visual Studio Code](https://code.visualstudio.com/docs/languages/python)）打开该文件夹。

在 rag_gpt_serp_scraping 目录下，新建一个空的 app.py 文件，用来存放你的抓取和 RAG 逻辑。

接下来，在项目目录下初始化一个 [Python 虚拟环境](https://docs.python.org/3/library/venv.html)：

```bash
python3 -m venv env
```

使用以下命令激活虚拟环境：

```bash
source ./env/bin/activate
```

### 步骤 #2：安装所需的库

本 Python RAG 项目将使用以下依赖：

- [`python-dotenv`](https://pypi.org/project/python-dotenv/): 用于安全地管理敏感凭据（例如 Bright Data 凭据和 OpenAI API 密钥）。  
- [`requests`](https://pypi.org/project/requests/): 用于向 Bright Data 的 SERP API 发起 HTTP 请求。  
- [`langchain-community`](https://pypi.org/project/langchain-community/): 用于从 Google SERP 返回的页面中获取文本，并进行清洗，从而为 RAG 生成相关内容。  
- [`openai`](https://pypi.org/project/openai/): 用于与 GPT 模型交互，从输入和 RAG 上下文中生成自然语言回复。  
- [`streamlit`](https://pypi.org/project/streamlit/): 用于创建一个简单的 UI，以便用户可以输入 Google 搜索关键词和 AI prompt 并动态查看结果。

安装所有依赖：

```bash
pip install python-dotenv requests langchain-community openai streamlit
```

我们将使用 [AsyncChromiumLoader](https://python.langchain.com/docs/integrations/document_loaders/async_chromium/)（来自 langchain-community），它需要以下依赖：

```bash
pip install --upgrade --quiet playwright beautifulsoup4 html2text
```

Playwright 还需要安装浏览器才能正常工作：

```bash
playwright install
```

### 步骤 #3：准备项目

在 `app.py` 中添加以下导入：

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st
```

然后，在项目文件夹中新建一个 `.env` 文件，用于存放所有凭据信息。现在你的项目结构大致如下图所示：

![Project structure](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-19.png)

在 `app.py` 中使用以下函数来告诉 `python-dotenv` 从 `.env` 文件中加载环境变量：

```python
load_dotenv()
```

之后，你就可以使用下面的语句从 `.env` 或系统中读取环境变量：

```python
os.environ.get("<ENV_NAME>")
```

### 步骤 #4：配置 SERP API

我们将使用 Bright Data 的 SERP API 来获取搜索引擎结果页信息，并在 Python RAG 工作流中使用这些信息。具体来说，我们会从 SERP API 返回的页面 URL 中提取文本。

要配置 SERP API，请参考 [官方文档](https://docs.brightdata.com/scraping-automation/serp-api/quickstart)。或者，按照下述说明进行操作。

如果你还没有创建账号，请先在 [Bright Data](https://www.bright.cn) 注册。登录后，进入账号控制台：

![Account main dashboard](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-18.png)

点击 “Get proxy products” 按钮。

随后会跳转到下图所示页面，点击 “SERP API” 对应一行：

![Clicking on SERP API](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-17.png)

在 SERP API 产品页中，切换 “Activate zone” 开关来启用该产品：

![Activating the SERP zone](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-16.png)

然后在 “Access parameters” 区域复制 SERP API 的 host、port、username 和 password，将它们添加到 `.env` 文件中：

```python
BRIGHT_DATA_SERP_API_HOST="<YOUR_HOST>"

BRIGHT_DATA_SERP_API_PORT=<YOUR_PORT>

BRIGHT_DATA_SERP_API_USERNAME="<YOUR_USERNAME>"

BRIGHT_DATA_SERP_API_PASSWORD="<YOUR_PASSWORD>"
```

将 `<YOUR_XXXX>` 占位符替换成 Bright Data 在 SERP API 页面上给出的实际值。

注意，此处 “Access parameters” 中的 host 格式类似于：

```python
brd.superproxy.io:33335
```

需要将其拆分为：

```python
BRIGHT_DATA_SERP_API_HOST="brd.superproxy.io"

BRIGHT_DATA_SERP_API_PORT=33335
```

### 步骤 #5：实现 SERP 数据抓取逻辑

在 `app.py` 中添加以下函数，用于获取 Google SERP 第一页的前 `number_of_urls` 个结果链接：

```python
def get_google_serp_urls(query, number_of_urls=5):
    # 使用 Bright Data 的 SERP API 发起请求
    # 并获取自动解析后的 JSON 数据

    host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")
    port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")
    username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")
    password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

    proxy_url = f"http://{username}:{password}@{host}:{port}"
    proxies = {"http": proxy_url, "https": proxy_url}

    url = f"https://www.google.com/search?q={query}&brd_json=1"
    response = requests.get(url, proxies=proxies, verify=False)

    # 获取解析后的 JSON 响应
    response_data = response.json()

    # 从响应中提取前 number_of_urls 个 Google SERP URL
    google_serp_urls = []

    if "organic" in response_data:
        for item in response_data["organic"]:
            if "link" in item:
                google_serp_urls.append(item["link"])

    return google_serp_urls[:number_of_urls]
```

以上代码会向 SERP API 发起一个 HTTP GET 请求，其中包含搜索词 query 参数。通过设置 [`brd_json=1`](https://docs.brightdata.com/scraping-automation/serp-api/parsing-search-results)，SERP API 会将搜索结果自动解析为 JSON 格式，类似如下结构：

```json
{
  "general": {
    "search_engine": "google",
    "results_cnt": 1980000000,
    "search_time": 0.57,
    "language": "en",
    "mobile": false,
    "basic_view": false,
    "search_type": "text",
    "page_title": "pizza - Google Search",
    "code_version": "1.90",
    "timestamp": "2023-06-30T08:58:41.786Z"
  },
  "input": {
    "original_url": "https://www.google.com/search?q=pizza&brd_json=1",
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/608.2.11 (KHTML, like Gecko) Version/13.0.3 Safari/608.2.11",
    "request_id": "hl_1a1be908_i00lwqqxt1"
  },
  "organic": [
    {
      "link": "https://www.pizzahut.com/",
      "display_link": "https://www.pizzahut.com",
      "title": "Pizza Hut | Delivery & Carryout - No One OutPizzas The Hut!",
      "image": "omitted for brevity...",
      "image_alt": "pizza from www.pizzahut.com",
      "image_base64": "omitted for brevity...",
      "rank": 1,
      "global_rank": 1
    },
    {
      "link": "https://www.dominos.com/en/",
      "display_link": "https://www.dominos.com › ...",
      "title": "Domino's: Pizza Delivery & Carryout, Pasta, Chicken & More",
      "description": "Order pizza, pasta, sandwiches & more online for carryout or delivery from Domino's. View menu, find locations, track orders. Sign up for Domino's email ...",
      "image": "omitted for brevity...",
      "image_alt": "pizza from www.dominos.com",
      "image_base64": "omitted for brevity...",
      "rank": 2,
      "global_rank": 3
    }
    // 省略...
  ],
  // 省略...
}
```

最后几行分析 JSON 数据并从中选取前 `number_of_urls` 个 SERP 结果链接并返回列表。

### 步骤 #6：从 SERP URL 中提取文本

定义一个函数，用于从获取的 SERP URL 中提取文本：

```python
# 注意：有些网站包含动态内容或反爬虫机制，可能导致文本无法提取。
# 如遇到此类问题，可考虑使用其它工具，比如 Selenium。
def extract_text_from_urls(urls, number_of_words=600): 
    # 指示一个无头 Chrome 实例访问给定的 URLs
    # 并使用指定的 user-agent

    loader = AsyncChromiumLoader(
        urls,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    )
    html_documents = loader.load()

    # 使用 BeautifulSoupTransformer 处理抓取到的 HTML 文档，从中提取文本
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html_documents,
        tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],
        unwanted_tags=["a"],
        remove_comments=True,
    )

    # 确保每个 HTML 文档仅保留 number_of_words 个单词
    extracted_text_list = []
    for doc_transformed in docs_transformed:
        # 将文本切分成单词，仅取前 number_of_words 个
        words = doc_transformed.page_content.split()[:number_of_words]
        extracted_text = " ".join(words)

        # 略过内容为空的文本
        if len(extracted_text) != 0:
            extracted_text_list.append(extracted_text)

    return extracted_text_list
```

该函数将会：

1. 使用无头 Chrome 浏览器实例访问传入的 URLs。  
2. 利用 [BeautifulSoupTransformer](https://python.langchain.com/v0.2/api_reference/community/document_transformers/langchain_community.document_transformers.beautiful_soup_transformer.BeautifulSoupTransformer.html) 处理每个页面的 HTML，以从特定标签（如 `<p>`、`<h1>`、`<strong>` 等）中提取文本，跳过不需要的标签（如 `<a>`）及注释。  
3. 对每个页面的文本只保留指定的单词数（`number_of_words`）。  
4. 返回一个含有每个 URL 所提取文本的列表。

对于某些特殊场景，你可能需要调整要提取的 HTML 标签列表，或者增加/减少需要保留的单词数。例如，假设我们对如下页面 [Transformers One 影评](https://athomeinhollywood.com/2024/09/19/transformers-one-review/) 应用此函数：

![Transformers one review page](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-15.png)

执行 `extract_text_from_urls()` 后得到的文本列表示例：

```python
["Lisa Johnson Mandell’s Transformers One review reveals the heretofore inconceivable: It’s one of the best animated films of the year! I never thought I’d see myself write this about a Transformers movie, but Transformers One is actually an exceptional film! ..."]
```

`extract_text_from_urls()` 返回的文本列表将用于为 OpenAI 模型提供 RAG 上下文。

### 步骤 #7：生成 RAG Prompt

定义一个函数，用于将 AI 请求（prompt）与文本上下文拼接成最后用于 RAG 的 prompt：

```python
def get_openai_prompt(request, text_context=[]):
    # 默认 prompt
    prompt = request

    # 如果有传入上下文，则将上下文与 prompt 拼接
    if len(text_context) != 0:
        context_string = "\n\n--------\n\n".join(text_context)
        prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

    return prompt
```

如果指定了 RAG 上下文，上面这个函数返回的 Prompt 大致如下格式：

```
Answer the request using only the context below.

Context:

Bla bla bla...

--------

Bla bla bla...

--------

Bla bla bla...

Request: <YOUR_REQUEST>
```

### 步骤 #8：执行 GPT 请求

首先，在 `app.py` 顶部初始化 OpenAI 客户端：

```python
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

其中 `OPENAI_API_KEY` 存储于环境变量，可直接定义在系统环境变量或 `.env` 文件中：

```
OPENAI_API_KEY="<YOUR_API_KEY>"
```

将其中的 `<YOUR_API_KEY>` 替换为你的 [OpenAI API key](https://platform.openai.com/api-keys) 值。如果需要创建或查找该密钥，请参考 [官方指南](https://platform.openai.com/docs/quickstart)。

然后，编写一个函数通过 OpenAI 官方客户端向 [gpt-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 模型发送请求：

```python
def interrogate_openai(prompt, max_tokens=800):
    # 使用给定 prompt 向 OpenAI 模型发送请求
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

> **注意**：  
> 你也可以使用 OpenAI API 提供的其他任意 GPT 模型。

如果在调用时传入的 prompt 包含 `get_openai_prompt()` 拼接的文本上下文，那么 `interrogate_openai()` 就能按我们的需求执行检索增强式生成（RAG）。

### 步骤 #9：创建应用程序的 UI

使用 Streamlit 来定义一个简单的 [Form UI](https://docs.streamlit.io/develop/concepts/architecture/forms)，让用户能够输入：

1. 用于 SERP API 的搜索关键词  
2. 想要发送给 GPT-4o mini 的 AI prompt  

示例如下：

```python
with st.form("prompt_form"):
    # 初始化输出结果
    result = ""
    final_prompt = ""

    # 让用户输入他们的 Google 搜索词
    google_search_query = st.text_area("Google Search:", None)

    # 让用户输入他们的 AI prompt
    request = st.text_area("AI Prompt:", None)

    # 提交按钮
    submitted = st.form_submit_button("Send")

    # 如果表单被提交
    if submitted:
        # 从给定搜索词中获取 Google SERP URLs
        google_serp_urls = get_google_serp_urls(google_search_query)

        # 从相应 HTML 页面中提取文本
        extracted_text_list = extract_text_from_urls(google_serp_urls)

        # 使用提取到的文本作为上下文生成 AI prompt
        final_prompt = get_openai_prompt(request, extracted_text_list)

        # 调用 OpenAI 模型进行询问
        result = interrogate_openai(final_prompt)

        # 展示生成后的完整 Prompt
        final_prompt_expander = st.expander("AI Final Prompt:")
        final_prompt_expander.write(final_prompt)

        # 输出来自 OpenAI 模型的回复
        st.write(result)
```

至此，我们的 Python RAG 脚本就完成了。

### 步骤 #10：整合所有部分

你的 `app.py` 文件整体应如下所示：

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st

# 加载 .env 文件中的环境变量
load_dotenv()

# 使用你的 API 密钥初始化 OpenAI 客户端
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_google_serp_urls(query, number_of_urls=5):
    # 使用 Bright Data 的 SERP API 发起请求
    # 并获取自动解析后的 JSON 数据
    host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")
    port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")
    username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")
    password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

    proxy_url = f"http://{username}:{password}@{host}:{port}"
    proxies = {"http": proxy_url, "https": proxy_url}

    url = f"https://www.google.com/search?q={query}&brd_json=1"
    response = requests.get(url, proxies=proxies, verify=False)

    # 获取解析后的 JSON 响应
    response_data = response.json()

    # 从响应中提取前 number_of_urls 个 Google SERP URL
    google_serp_urls = []

    if "organic" in response_data:
        for item in response_data["organic"]:
            if "link" in item:
                google_serp_urls.append(item["link"])

    return google_serp_urls[:number_of_urls]

def extract_text_from_urls(urls, number_of_words=600):
    # 指示一个无头 Chrome 实例访问给定的 URLs
    # 并使用指定的 user-agent
    loader = AsyncChromiumLoader(
        urls,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    )
    html_documents = loader.load()

    # 使用 BeautifulSoupTransformer 处理抓取到的 HTML 文档，从中提取文本
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html_documents,
        tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],
        unwanted_tags=["a"],
        remove_comments=True,
    )

    # 确保每个 HTML 文档仅保留 number_of_words 个单词
    extracted_text_list = []
    for doc_transformed in docs_transformed:
        words = doc_transformed.page_content.split()[:number_of_words]
        extracted_text = " ".join(words)
        if len(extracted_text) != 0:
            extracted_text_list.append(extracted_text)

    return extracted_text_list

def get_openai_prompt(request, text_context=[]):
    # 默认 prompt
    prompt = request

    # 如果有传入上下文，则将上下文与 prompt 拼接
    if len(text_context) != 0:
        context_string = "\n\n--------\n\n".join(text_context)
        prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

    return prompt

def interrogate_openai(prompt, max_tokens=800):
    # 使用给定 prompt 向 OpenAI 模型发送请求
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# 创建 Streamlit 表单，供用户输入
with st.form("prompt_form"):
    # 初始化输出结果
    result = ""
    final_prompt = ""

    # 让用户输入 Google 搜索词
    google_search_query = st.text_area("Google Search:", None)

    # 让用户输入 AI Prompt
    request = st.text_area("AI Prompt:", None)

    # 提交按钮
    submitted = st.form_submit_button("Send")

    if submitted:
        # 获取搜索词对应的 Google SERP URL 列表
        google_serp_urls = get_google_serp_urls(google_search_query)

        # 从对应的 HTML 页面中提取文本
        extracted_text_list = extract_text_from_urls(google_serp_urls)

        # 使用提取到的文本作为上下文生成最终的 Prompt
        final_prompt = get_openai_prompt(request, extracted_text_list)

        # 调用 OpenAI 模型获取结果
        result = interrogate_openai(final_prompt)

        # 展示生成后的完整 Prompt
        final_prompt_expander = st.expander("AI Final Prompt")
        final_prompt_expander.write(final_prompt)

        # 输出来自 OpenAI 的结果
        st.write(result)
```

### 步骤 #11：测试应用程序

使用以下命令运行你的 Python RAG 应用：

```bash
# 注意：Streamlit 在轻量级应用场景非常方便，但如果要用于生产环境，
# 可以考虑使用 Flask 或 FastAPI 等更适合生产部署的框架。
streamlit run app.py
```

在终端里，你应该会看到类似如下输出：

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://172.27.134.248:8501
```

根据提示，在浏览器中打开 `http://localhost:8501`。你会看到如下界面：

![Streamlit app screenshot](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-14.png)

可以尝试输入如下一条搜索：

```
Transformers One review
```

以及如下 AI prompt：

```
Write a review for the movie Transformers One
```

点击 “Send”，等待应用处理请求。数秒后，你就能看到类似下图的结果：

![App result screenshot](https://github.com/bright-cn/rag-chatbot/blob/main/Images/image-13.png)

若展开 “AI Final Prompt” 下拉框，你会看到应用为 RAG 生成的完整 Prompt。

## 结论

在使用 Python 来构建 RAG 聊天机器人时，主要挑战在于如何抓取像 Google 这样的搜索引擎：

1. 它们会频繁修改 SERP 页面的结构。  
2. 它们拥有十分复杂的反爬虫机制。  
3. 并发大规模获取 SERP 数据成本高且实现困难。

[Bright Data 的 SERP API](https://www.bright.cn/products/serp-api) 可以帮助你轻松地从各大搜索引擎获取实时 SERP 数据，同时也支持 RAG 以及许多其他应用场景。现在就开始你的免费试用吧！
