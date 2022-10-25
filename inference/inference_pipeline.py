import os
import argparse
import json

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers
import sys

sys.path.append("..")


class BiEncoder_GPT:
    def __init__(
        self,
        retrieval_model,
        generative_model,
    ):
        self.retrieval_model = retrieval_model
        self.generative_model = generative_model

    def calculate_candidats(self, candidats_texts) -> torch.Tensor:
        """расчитывает векторную базу кандидатов

        Args:
            candidats_texts (list[str]): список фактов о персоне
        Returns:
            torch.Tensor: вектора фактов о персоне
        """
        candidats_tokens = self.retrieval_model.collator.CandidateCollator(
            candidats_texts
        )
        candidats_vec = self.retrieval_model.encode_candidats(candidats_tokens)
        return candidats_vec

    def retrieve_gk(
        self,
        context_texts,
        candidats_vecs,
        top_k=3,
        th=0,
    ):
        """находит релевантные контексту кандидатов

        Args:
            context_texts (list(str)): текст сообщения к которому ищем кандидата
            candidats_vecs (list(torch.Tensor)): вектора фактов о персоне

        Returns:
            tuple(list(int), list(float)), : 0 -индексы соответсвующие номеру факта из персоны, 1 - растояние ко всем кандидатам
        """
        context_texts = context_texts[-1][1]
        context_tokens = self.retrieval_model.collator.ContextCollator(
            [[context_texts]]
        )
        context_vec = self.retrieval_model.encode_context(context_tokens)
        candidats_vecs = torch.tensor(candidats_vecs)
        context_vec = context_vec.repeat(candidats_vecs.size()[0], 1)
        distances = self.retrieval_model.compute_sim(context_vec, candidats_vecs)[0]
        candidats = distances * (distances > th)
        candidats = torch.topk(candidats, top_k).indices
        return candidats.tolist(), distances.tolist()

    def generate_reply(self, context_texts, gks) -> tuple:
        """генерирует ответ модели

        Args:
            context_texts (list): содержит tuple где 0 элемент 'model'/'user', 1 элемент - текст сообщения [('user', 'text'), ...]
            gks (list): содержит релевантные gk ['fact1', ...]

        Returns:
            tuple(str, [list(str)]): 0 - сгенерированное сообщение, 1 - новые сгенерированные gk
        """
        # TODO: расширить регулярки
        context_texts = [i[1] for i in context_texts]
        print("gks", gks)
        dict_inp = [{"context": context_texts, "gk": gks, "candidate": ""}]
        gpt_inp = self.generative_model.collator.test(dict_inp)[0]["input_ids"][:, :-2]
        print("gpt_inp", self.generative_model.tokenizer.batch_decode(gpt_inp))
        gpt_out = self.generative_model.GPT.generate(
            gpt_inp,
            max_new_tokens=64,
        )
        gpt_out = self.generative_model.tokenizer.decode(
            gpt_out[0][-64:], skip_special_tokens=False
        )
        print(gpt_out)
        gpt_out = gpt_out.split("[Gk]")
        msg = (
            gpt_out[-1].split("[P2u]")[1].split("[P1u]")[0].replace("<|endoftext|>", "")
        )
        new_gks = gpt_out[:-1]
        return msg, new_gks
