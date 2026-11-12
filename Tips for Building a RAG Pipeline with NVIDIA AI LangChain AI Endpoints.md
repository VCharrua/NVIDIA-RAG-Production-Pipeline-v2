**Tips for Building a RAG Pipeline with NVIDIA AI LangChain AI Endpoints**

**AI-Generated Summary**

- Retrieval-augmented generation (RAG) combines information retrieval with system prompts to provide more accurate, up-to-date, and contextually relevant responses from large language models (LLMs) by incorporating data from sources such as relational databases, document repositories, and news feeds.
- The post demonstrates building a RAG pipeline using [NVIDIA AI Endpoints for LangChain](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints), creating a vector store with the NVIDIA NeMo Retriever embedding microservice and FAISS for similarity search.
- Using NVIDIA Triton Inference Server documentation as the knowledge source, the pipeline shows how RAG eliminates hallucinations and delivers technically detailed answers compared to a vanilla LLM baseline.
- Optimizing chunk size for document splitting is critical for RAG performance, with typical values of 100-600 tokens depending on the LLM context window, and ensemble retrieval with mean reciprocal rank (MRR) benchmarking helps identify the optimal size.
- LangChain evaluators including Criteria Evaluator, Embedding Distance Evaluator, and Exact Match Evaluator enable measuring correctness, relevance, embedding similarity, and exact matches to ensure RAG systems are accurate, relevant, and free from hallucinations.

**Next Steps**

- Follow the [Build a RAG chain by generating embeddings for NVIDIA Triton Inference Server documentation](https://nvidia.github.io/GenerativeAIExamples/latest/notebooks/10_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.html) notebook to implement the pipeline.
- Explore the [full notebook](https://github.com/NVIDIA/GenerativeAIExamples/blob/main/RAG/notebooks/langchain/RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.ipynb) in the NVIDIA Generative AI Examples repository for complete code.
- Review [NVIDIA AI LangChain endpoints](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints) for additional models and chain configurations.

[Retrieval-augmented generation](https://www.nvidia.com/en-us/glossary/retrieval-augmented-generation/) (RAG) is a technique that combines information retrieval with a set of carefully designed system prompts to provide more accurate, up-to-date, and contextually relevant responses from [large language models](https://www.nvidia.com/en-us/glossary/large-language-models/) (LLMs). By incorporating data from various sources such as relational databases, unstructured document repositories, internet data streams, and media news feeds, RAG can significantly improve the value of [generative AI](https://www.nvidia.com/en-us/glossary/generative-ai/) systems.

Developers must consider a variety of factors when building a RAG pipeline: from LLM response benchmarking to selecting the right chunk size.

In this post, I demonstrate how to build a RAG pipeline using [NVIDIA AI Endpoints for LangChain](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints). First, you create a vector store by downloading web pages and generating their embeddings using the [NVIDIA NeMo Retriever embedding microservice](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-40k) and searching for similarity using [FAISS](https://github.com/facebookresearch/faiss). I then showcase two different chat chains for querying the vector store. For this example, I use the [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), though the code can be easily modified to use any other source.

For more information and to follow along, see the [Build a RAG chain by generating embeddings for NVIDIA Triton Inference Server documentation](https://nvidia.github.io/GenerativeAIExamples/latest/notebooks/10_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.html) notebook.

**Tutorial prerequisites**

To make the best use of this tutorial, you need basic knowledge of [LLM training](https://developer.nvidia.com/blog/mastering-llm-techniques-training/) and inference pipelines along with the following resources:

- [LangChain](https://python.langchain.com/)
- [NVIDIA AI Foundation Endpoints](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints)
- A vector store

**What is RAG and how does it empower LLMs?**

Here's why it's important to augment LLMs with a RAG pipeline.

Under the hood, LLMs are neural networks, typically measured by how many parameters they contain. An LLM's parameters essentially represent the general patterns of how humans use words to form sentences.

That deep understanding, sometimes called _parameterized knowledge_, makes LLMs useful in responding to general prompts at light speed. However, it does not serve users who want a deeper dive into a current or more specific topic.

RAG fills a gap in how LLMs work (Figure 1). Its purpose is to link generative AI services to external resources, especially ones rich in the latest technical details. The [original paper](https://arxiv.org/pdf/2005.11401.pdf) called RAG "a general-purpose fine-tuning recipe" because it can be used by nearly any LLM to connect with practically any external resource.

**Reduce LLM hallucinations and improve model responses**

RAG offers numerous benefits in the realm of language model technology, particularly when it comes to empowering LLMs with up-to-date information. By integrating a search function that retrieves relevant data from external sources, RAG ensures that LLMs are equipped with the most current knowledge available, thereby enhancing the accuracy and relevance of their responses (Figure 2).

RAG also provides a solution to the challenge of data privacy, as it enables LLMs to generate responses without requiring direct access to sensitive data. This is achieved by retrieving only the necessary information from external sources, thereby minimizing the risk of data breaches.

RAG has the potential to mitigate the issue of LLM hallucinations, which refers to the generation of inaccurate or misleading information due to the limitations of the model's training data. By providing LLMs with real-time access to external sources, RAG can help reduce the likelihood of hallucinations and improve the overall reliability of the model's responses.

**Considerations for RAG implementation**

RAG presents some challenges that must be addressed to fully realize its potential. One such challenge is ensuring that the prompt used to retrieve information from external sources is well-crafted and accurately reflects the user's intent. A poorly constructed prompt may result in irrelevant or incomplete information being retrieved, which can negatively impact the quality of the LLM's response.

Another challenge is determining how to evaluate the success of RAG in a meaningful and objective manner. Metrics such as response accuracy and relevance are important, but they may not fully capture the nuances of RAG's impact on LLM performance.

Finally, optimizing RAG to maximize its benefits and minimize its challenges is an ongoing process that requires careful consideration of factors such as search algorithm efficiency, information retrieval relevance, and LLM integration. For more information about other challenges, see the [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856) paper and [12 RAG Pain Points and Proposed Solutions](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c) article.

By addressing these challenges, RAG has the potential to significantly enhance the capabilities of LLMs and unlock new possibilities for natural language processing applications.

**What is LangChain used for?**

[LangChain](https://www.langchain.com/) is an open-source framework that simplifies the development of applications using LLMs. It provides tools and abstractions to improve the customization, accuracy, and relevancy of LLMs, enabling the creation of various applications like chatbots, question-answering, content generation, and summarizers.

LangChain consists of LangChain Libraries, LangChain Templates, LangServe, and LangSmith, offering interfaces, integrations, reference architectures, and a developer platform for building and deploying LLM-powered applications. The framework includes standard interfaces for Model I/O, Retrieval, and Agents, enabling you to integrate data sources, LLMs, and tools to build complex applications.

LangChain is part of a rich ecosystem of tools, supported by an active community, and simplifies AI development by abstracting the complexity of data source integrations.

**Set up**

To get started, create a free account with the [NVIDIA API catalog](https://build.nvidia.com/) and follow these steps:

1. Select any model.
2. Choose **Python**, **Get API Key**_._
3. Save the generated key as NVIDIA_API_KEY.

From there, you should have access to the endpoints.

The [full notebook for this post](https://github.com/NVIDIA/GenerativeAIExamples/blob/main/RAG/notebooks/langchain/RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.ipynb) is available as part of the [NVIDIA Generative AI Examples](https://nvidia.github.io/GenerativeAIExamples/latest) GitHub repo.

First, install LangChain, NVIDIA AI Endpoints, and FAISS.

```
pip install langchain
pip install langchain_nvidia_ai_endpoints
pip install faiss-gpu
```

**Baseline**

To establish a baseline for comparison, first evaluate the response using a vanilla LLM:

```
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="ai-llama2-70b", max_tokens=1000)
```

Now try asking a basic question about NVIDIA Triton Inference Server.

```
result = llm.invoke("What interfaces does Triton support?")
print(result.content)
```

Triton, a product offering from Cohesity, is a data protection and management solution designed to simplify and streamline the backup and recovery of data across various environments. Triton supports several interfaces, including…

The baseline LLM does not recognize Triton Inference Server within the context of the question and responds with incorrect information. Try adding NVIDIA to improve the prompt.

```
result = llm.invoke("What interfaces does NVIDIA Triton support?")
print(result.content)
```

NVIDIA Triton is a platform for AI model inference serving, and it supports several interfaces to provide flexibility and ease of integration with various applications and frameworks.

It now recognizes Triton Inference Server, but doesn't give much detail.

```
result = llm.invoke("But why?")
print(result.content)
```

These interfaces ensure that NVIDIA Triton can be easily integrated into various AI and machine learning projects, providing a scalable and high-performance inference serving solution.

I follow these guidelines to ensure that I provide valuable, safe, and respectful assistance to you. As a helpful, reliable, and trustworthy assistant, I am committed to promoting positive and fair interactions. By avoiding harmful, unethical, prejudiced, or negative content, I aim to create a helpful and secure environment for all users.

Looking at the second sentence, you see that the LLM hallucinates. Now, add external sources of information to the LLM to augment it for your prompts and evaluate whether it improves the accuracy of the responses.

**Read HTML and split text**

Load a list of web pages containing technical documentation about [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) in preparation for the embedding vector store.

<table><tbody><tr><th><pre><code># List of web pages containing NVIDIA Triton technical documentation
&nbsp;&nbsp;&nbsp;&nbsp;urls = [&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "<a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html">https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html</a>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"<a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html">https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html</a>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"<a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html">https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html</a>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"<a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html">https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html</a>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"<a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html">https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html</a>",
&nbsp;&nbsp;&nbsp;&nbsp;]</code></pre><p>&nbsp;</p><pre><code>&nbsp;&nbsp;&nbsp;&nbsp;documents = []
&nbsp;&nbsp;&nbsp;&nbsp;for url in urls:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;document = html_document_loader(url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;documents.append(document)</code></pre></th></tr></tbody></table>

**Split into chunks**

Next, split the documents into separate chunks. Make sure to pay attention to the chunk_size parameter in TextSplitter. Setting the right chunk size is critical for RAG performance, as much of a RAG pipeline's success is based on the retrieval step finding the right context for generation.

The retrieval step typically examines smaller chunks of the original text rather than all documents. The entire prompt (retrieved chunks plus the user query) must fit within the LLM's context window. Don't specify chunk sizes too big and do balance them out with the estimated query size.

For example, while OpenAI LLMs have a context window of 8–32K tokens, Llama2 is limited to 4K tokens. If the chunks are too small, there is a risk that vital information might not be among the top retrieved chunks due to high granularity. On the other hand, if the chunks are too big, they may not fit in the LLM context window, slowing down the system.

To address this, build an ensemble retrieval over different chunk sizes and benchmark the results to find the optimal value. By looping the ensemble on a set of test queries, you can calculate the mean reciprocal rank (MRR) for each chunk size, for a more informed decision on the optimal chunk size for the RAG system.

Experiment with different chunk sizes, but typical values should be 100-600, depending on the LLM.

```
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
```

**Generate embeddings**

Next, generate embeddings using [NVIDIA AI Foundation endpoints](https://python.langchain.com/docs/integrations/text_embedding/nvidia_ai_endpoints) and save embeddings to an offline vector store in the /embed directory for future re-use.

For this task, use [FAISS](https://github.com/facebookresearch/faiss), a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

<table><tbody><tr><th><pre><code>embeddings = NVIDIAEmbeddings()</code></pre><p>&nbsp;</p><pre><code>&nbsp;&nbsp;&nbsp;&nbsp;for document in documents:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;texts = splitter.split_text(document.page_content)</code></pre><p>&nbsp;</p><pre><code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# metadata to attach to document
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;metadatas = [document.metadata]</code></pre><p>&nbsp;</p><pre><code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# create embeddings and add to vector store
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if os.path.exists(dest_embed_dir):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update.add_texts(texts, metadatas=metadatas)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update.save_local(folder_path=dest_embed_dir)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docsearch.save_local(folder_path=dest_embed_dir)</code></pre></th></tr></tbody></table>

**Load embeddings from the vector store**

Create the embeddings model using the [NVIDIA NeMo Retriever text embedding model](https://build.nvidia.com/nvidia/embed-qa-4). This model represents words, phrases, or other entities as vectors of numbers and understands the relation between words and phrases.

```
embedding_model = NVIDIAEmbeddings()
embedding_path = "embed/"
docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model)
```

First, create a [ConversationalRetrievalChain](https://python.langchain.com/docs/modules/chains/popular/chat_vector_db#conversationalretrievalchain-with-streaming-to-stdout) chain using ChatNVIDIA. In this chain, I demonstrate the use of one LLM.

<table><tbody><tr><th><pre><code>llm = ChatNVIDIA(model="ai-llama2-70b", temperature=0.1, max_tokens=1000, top_p=1.0)</code></pre><p>&nbsp;</p><pre><code>qa_prompt=QA_PROMPT</code></pre><p>&nbsp;</p><pre><code>doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)</code></pre><p>&nbsp;</p><pre><code>qa = ConversationalRetrievalChain.from_llm(
&nbsp;&nbsp;&nbsp;&nbsp;llm=llm,
&nbsp;&nbsp;&nbsp;&nbsp;retriever=docsearch.as_retriever(),
&nbsp;&nbsp;&nbsp;&nbsp;chain_type="stuff",
&nbsp;&nbsp;&nbsp;&nbsp;memory=memory,
&nbsp;&nbsp;&nbsp;&nbsp;combine_docs_chain_kwargs={'prompt': qa_prompt},
)</code></pre></th></tr></tbody></table>

**Ask any question about Triton Inference Server**

Now try using the same prompts and gauging the difference.

```
query = "What interfaces does Triton support?"
result = qa({"question": query})
print(result.get("answer"))
```

NVIDIA Triton Inference Server supports several interfaces, including HTTP/REST and gRPC inference protocols. These interfaces are based on the community-developed KServe protocol. Additionally, Triton Inference Server provides a C API and Java API, which allows it to link directly into your application for edge and other in-process use cases. It also supports various deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, and RAPIDS FIL, among others.

The LLM recognizes Triton Inference Server correctly with a minimal prompt and responds with a technically detailed response.

```
query = "But why?"
result = qa({"question": query})
print(result.get("answer"))
```

Triton Inference Server supports various interfaces and frameworks to provide flexibility and ease of use for developers and data scientists. By supporting multiple deep learning and machine learning frameworks such as TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more, Triton Inference Server allows users to deploy any AI model that best suits their project requirements. This interoperability helps in streamlining the AI inferencing process across different platforms, including cloud, data center, edge, and embedded devices, and ensures optimized performance for various query types. Furthermore, Triton Inference Server supports HTTP/REST and gRPC inference protocols, which are widely used and recognized in the industry, making it a versatile solution for AI inference serving.

The LLM now no longer hallucinates as it has knowledge of the domain.

**A more complex chain**

Now create a more complex chain with two LLMs, one for summarization and another for chat. This improves the overall result in more complicated scenarios. Use Llama2 70B for the first LLM and Mixtral for the chat element in the chain. Add question_generator to generate relevant query prompts.

```
llm = ChatNVIDIA(model="ai-llama2-70b")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
chat = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)
doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)
qa = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(),
    combine_docs_chain=doc_chain,
    memory=memory,
    question_generator=question_generator,
)
```

Here's the next query and response:

```
query = "What interfaces does Triton support?"
result = qa({"question": query})
print(result.get("answer"))
```

The supported interfaces in Triton for inference are:

\* HTTP/REST \* GRPC \* Binary Tensor Data Extension \* Classification Extension \* Generate Extension \* Logging Extension \* Model Configuration Extension \* Model Repository Extension \* Schedule Policy Extension \* Sequence Extension \* Shared-Memory Extension \* Statistics Extension \* Trace Extension

These interfaces allow for a wide range of use cases, including real-time inference, batched inference, ensemble inference, and audio/video streaming. Additionally, Triton provides a Backend API that allows developers to add custom backends and pre/post-processing operations, as well as support for model pipelines using Ensembling or Business Logic Scripting (BLS).

The response is now summarized in bullets:

```
query = "But why?"
result = qa({"question": query})
print(result.get("answer"))
```

Triton supports many interfaces to provide flexibility and versatility for its users. By supporting multiple deep learning and machine learning frameworks, Triton allows developers to use their framework of choice, making it easier for them to integrate Triton into their existing workflows. This flexibility enables Triton to be used in a wide range of applications, from real-time inference to batch processing, and from cloud-based services to edge devices.

And the follow-up response is also more concise.

**How to evaluate**

Choosing the right metric for evaluating results is critical for RAG systems. It is not enough to gauge accuracy alone. Relevancy, an absence of hallucinations, and runtime metrics such as average response time must also be considered.

[LangChain](https://www.langchain.com/) includes [evaluators](https://python.langchain.com/docs/guides/evaluation/) for such metrics:

- [Criteria Evaluator](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain) checks correctness by assigning a score of 1-5 along with a reasoning for the score. It can also evaluate the relevance of a response considering the context.
- [Embedding Distance Evaluator](https://python.langchain.com/docs/guides/evaluation/string/embedding_distance) evaluates the embedding similarity of the response.
- [Exact Match Evaluator](https://python.langchain.com/docs/guides/evaluation/string/exact_match) compares the response to a reference label.

By using these metrics, you can ensure that your RAG systems are accurate, relevant, and free from hallucinations, while also providing fast and efficient responses.

**Conclusion**

RAG has emerged as a powerful approach, combining the strengths of LLMs and dense vector representations. By using dense vector representations, RAG models can scale efficiently, making them well-suited for large-scale enterprise applications.

As LLMs continue to evolve, it is clear that RAG will play an increasingly important role in driving innovation and delivering high-quality, intelligent systems that can understand and generate human-like language.

When building your own RAG pipeline, it's important to correctly split the vector store documents into chunks by optimizing the chunk size for your specific content and selecting a LLM with suitable context length. For some cases, complex chains of multiple LLMs may be required. To optimize RAG performance and measure success, use a collection of robust evaluators and metrics.

To get started, the [full notebook for this post](https://github.com/NVIDIA/GenerativeAIExamples/blob/main/RAG/notebooks/langchain/RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.ipynb) is available as part of the [NVIDIA Generative AI Examples repository](https://nvidia.github.io/GenerativeAIExamples/latest). For more information about additional models and chains, see [NVIDIA AI LangChain endpoints](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints).

**Tags**

[Agentic AI / Generative AI](https://developer.nvidia.com/blog/category/generative-ai/) | [Cloud Services](https://developer.nvidia.com/blog/recent-posts/?industry=Cloud+Services) | [AI Foundation Models](https://developer.nvidia.com/blog/recent-posts/?products=AI+Foundation+Models) | [NeMo Retriever](https://developer.nvidia.com/blog/recent-posts/?products=NeMo+Retriever) | [Intermediate Technical](https://developer.nvidia.com/blog/recent-posts/?learning_levels=Intermediate+Technical) | [Best practice](https://developer.nvidia.com/blog/recent-posts/?content_types=Best+practice) | [featured](https://developer.nvidia.com/blog/tag/featured/) | [LangChain](https://developer.nvidia.com/blog/tag/langchain/) | [Retrieval Augmented Generation (RAG)](https://developer.nvidia.com/blog/tag/retrieval-augmented-generation-rag/)

**About the Authors**

**About Amit Bleiweiss**  
Amit Bleiweiss is a senior data scientist at NVIDIA, where he focuses on large language models and generative AI. He has 25 years of experience in applied machine learning and deep learning, with over 50 patents and publications in the domain. Amit received his MSc from Hebrew University of Jerusalem, where he specialized in machine learning.

[**View all posts by Amit Bleiweiss**](https://developer.nvidia.com/blog/author/ambleiweiss/)

**Comments**

Notable Replies

1. [duenas.marta](https://forums.developer.nvidia.com/t/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/292227)says:

May 30, 2024

Good morning,  
I am trying RAG implementation.  
I follow the instructions on your [blog](https://developer.nvidia.com/blog/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/)  
I can't connect to: **from langchain_nvidia_ai_endpoints import ChatNVIDIA**  
I only get it as specified by "NVIDIA API catalog", through **from openai import OpenAI**

If I make the invocation with ChatNVIDIA and the url they provide me in "NVIDIA API catalog"  
_from langchain_nvidia_ai_endpoints import ChatNVIDIA_  
_llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=nvapi_key)_  
_llm.invoke("What interfaces does Triton support?")_  
_print(result.content)_  
answer: "ValueError: Unknown model name meta/llama3-70b-instruct specified." available models ai-llama3-70b - a88f115a-4a47-4381-ad62-ca25dc33dc1b …  
If I make the invocation with model="ai-llama3-70b"  
_llm = ChatNVIDIA(model="ai-llama3-70b", nvidia_api_key=nvapi_key)_  
answer: Exception: \[404\] Not Found \\n The model gpt43b does not exist.

2. [ambleiweiss](https://forums.developer.nvidia.com/t/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/292227)says:

May 30, 2024

Note the blog uses the model "ai-llama2-70b"

You can follow a full notebook version of the blog here: [Build a RAG chain by generating embeddings for NVIDIA Triton documentation — NVIDIA Generative AI Examples 24.4.0 documentation](https://nvidia.github.io/GenerativeAIExamples/latest/notebooks/10_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.html)

In order to see the supported models, call _ChatNVIDIA.get_available_models()_

3. [ashish-kumar](https://forums.developer.nvidia.com/t/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/292227)says:

July 24, 2024

Hi - I am facing issue with langchain using ChatNVIDIA with locally deployed LLM with NIM.

My Code:  
from langchain_nvidia_ai_endpoints import ChatNVIDIA  
llm = ChatNVIDIA(  
base_url="[https://ashish-mistral-nim-deploy-1-predictor](https://ashish-mistral-nim-deploy-1-predictor/).\*\*\*\*\*\*\*\*\*\*/v1/completions",  
model="mistralai/mistral-7b-instruct-v0.3"  
)  
result = llm.invoke("Write a ballad about LangChain.")

This code is throwing below error  
SSLError: HTTPSConnectionPool(host='ashish-mistral-nim-deploy-1-predictor.\*\*\*\*\*\*\*\*\*\*\*', port=443): Max retries exceeded with url: /v1/completions/chat/completions (Caused by SSLError(SSLCertVerificationError(1, '\[SSL: CERTIFICATE_VERIFY_FAILED\] certificate verify failed: unable to get local issuer certificate (\_ssl.c:1006)')))

Am not sure how to apply the ssl certificate. Any guidaince here?

**Following curl based api request works perfectly fine though.**

curl --cacert test.crt -X 'POST' '[https://ashish-mistral-nim-deploy-1-predictor](https://ashish-mistral-nim-deploy-1-predictor/).\*\*\*\*\*\*\*\*\*\*/v1/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "mistralai/mistral-7b-instruct-v0.3","prompt": "Write a ballad about LangChain.","max_tokens": 64}'

4. [ambleiweiss](https://forums.developer.nvidia.com/t/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/292227)says:

July 25, 2024

Hi Ashish,

When running in LangChain you need to remove "/completions" from the base_url

See this line in the notebook example: llm = ChatNVIDIA(base_url="<http://0.0.0.0:8000/v1>", model="meta/llama3-8b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)

Thanks,  
Amit

5. [ashish-kumar](https://forums.developer.nvidia.com/t/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/292227)says:

July 25, 2024

Hi - Thanks for response. Your hints did not help here.  
Below API Call fails…  
llm = ChatNVIDIA(  
base_url="[https://ashish-mistral-nim-deploy-1-predictor](https://ashish-mistral-nim-deploy-1-predictor/).\*\*\*\*\*\*/v1",  
model="mistralai/mistral-7b-instruct-v0.3"  
)

Error  
SSLError: HTTPSConnectionPool(host='ashish-mistral-nim-deploy-1-\*\*\*\*\*\*\*\*\*', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLCertVerificationError(1, '\[SSL: CERTIFICATE_VERIFY_FAILED\] certificate verify failed: unable to get local issuer certificate (\_ssl.c:1006)')))

# Following piece of code works perfectly fine as well.

import requests

langchain_nvidia_mistral_api_url="<https://ashish-mistral-nim-deploy-1-\*\*\*\*\*\*\*/v1/completions>"  
tokenheaders={"Content-Type":"application/json", "accept": "application/json"}  
payload={"model": "mistralai/mistral-7b-instruct-v0.3","prompt": "Tell me something about Langchain.","max_tokens": 64}  
response = requests.post(langchain_nvidia_mistral_api_url, json=payload, headers=tokenheaders, verify=False)

print(response)  
print(response.json())

How do i pass my ssl ca certificate to langchain api call? or how can i mention verify=False in NVIDIAChat API Call? Just adding verify=False does not work.