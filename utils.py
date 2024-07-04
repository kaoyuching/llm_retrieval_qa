from typing import Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class QAChain():
    def __init__(
        self,
        llm_model,
        vector_db,
        prompt_template,
        top_k: int = 10,
        return_source_documents: bool = False,
        similarity_score_threshold: Optional[float] = None,
    ):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold

    def __call__(self, question):
        query_docs = self.vector_db.similarity_search_with_score(
            question,
            k=self.top_k,
            distance_strategy=DistanceStrategy.COSINE,
        )

        doc_str = []
        contexts = []
        scores = []
        for _doc, _score in query_docs:
            if self.threshold is None or _score <= self.threshold:
                doc_str.append(_doc)
                contexts.append(_doc.page_content)
            scores.append(_score)
        if len(contexts) == 0:
            contexts = "Found nothing in the documents"
        else:
            contexts = '\n'.join(contexts)
        input_prompt = self.prompt_template.format(context=contexts, question=question)

        res = self.llm_model.invoke(input_prompt)
        output = {'query': question, 'result': res}
        if self.return_source_documents:
            output['source_documents'] = doc_str
        return output
