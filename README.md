# Contextual Document Embeddings

This repository contains the notes made from the paper **"Contextual Document Embeddings"**.

## Abstract

Dense document embeddings are crucial for neural retrieval tasks. The dominant paradigm typically trains document encoders on individual documents without considering context. In this work, the authors argue that document embeddings should take into account both the document itself and its neighboring documents in context, similar to how contextualized word embeddings work. They propose two complementary methods for contextualized document embeddings that significantly improve retrieval performance, especially in out-of-domain settings.

## Introduction

Traditional dense document embeddings are constructed by encoding individual documents, making the embeddings effective but context-independent. This approach misses the opportunity to model relationships between a document and its surrounding context, which is important for targeted retrieval use cases.

We propose two methods to address this limitation:

1. **Contextual Contrastive Learning**: A new contrastive learning objective that incorporates neighboring documents into the intra-batch loss, enabling better contextualization of document embeddings.
2. **Contextual Architecture**: A new architecture that explicitly encodes information from neighboring documents, directly incorporating contextualized data into the document representation.

Both methods aim to generate document embeddings that are more contextually aware and thus more effective for retrieval tasks.

## Methods

### Contextual Contrastive Learning

The proposed contrastive learning objective differs from standard contrastive learning by explicitly integrating document neighbors into the loss calculation. This approach adjusts the traditional framework to better capture the context of a document in relation to other documents.

- **Intra-batch Contextual Loss**: Rather than treating each document in isolation, they compute a loss that encourages the model to learn representations that are contextually aware of neighboring documents within the same batch.
- This loss helps the model to optimize document representations that capture both the individual content and its relevance within the surrounding context.

### Contextual Architecture

In addition to the contrastive learning method, they propose a new architecture designed to encode neighboring document information directly into the document's representation. This architecture differs from the traditional biencoder approach by using cross-document context to enrich the embedding.

- **Neighbor Encoding**: Each document is encoded not only by itself but also by considering the context provided by neighboring documents. This results in a richer, more context-aware representation.
- The architecture works by combining both the individual document encoding and the contextual information from neighboring documents, producing a more comprehensive and useful document embedding.

## Results

Both methods significantly outperform traditional biencoder approaches, especially in out-of-domain scenarios. Their experiments show the following:

- **Improved Performance**: Both the contextual contrastive learning and the contextual architecture methods outperform biencoders in multiple retrieval settings, with particularly significant improvements observed in out-of-domain settings.
- **State-of-the-Art on MTEB**: Their method achieves state-of-the-art results on the **MTEB benchmark** without relying on complex techniques such as hard negative mining, score distillation, dataset-specific instructions, or extremely large batch sizes.
- The methods can be applied to any contrastive learning task and improve the performance of any biencoder model, demonstrating their general applicability.

## Conclusion

The **Contextual Document Embeddings** framework offers a novel approach to document retrieval by incorporating neighboring documents into the embedding process, resulting in more accurate and context-aware document representations. Their methods show superior performance over traditional biencoders, particularly in out-of-domain retrieval, and achieve state-of-the-art results without relying on complex strategies such as large batch sizes or hard negative mining.

## References

- Contextual Document Embeddings Paper : [Contextual Document Embeddings.pdf](https://github.com/user-attachments/files/18469743/Contextual.Document.Embeddings.pdf)
- Reimers, N., & Gurevych, I. (2020). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks*. arXiv:1908.10084.

