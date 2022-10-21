import os
import sys

sys.path.append("/home/stc/persona/")
from models import BERT_RetrievalModel, GPT_GenerativeModel
from inference_pipeline import BiEncoder_GPT
import logging
from telegram import Update
from telegram.ext import (
    filters,
    Application,
    ApplicationBuilder,
    MessageHandler,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ExtBot,
    TypeHandler,
    BaseHandler,
)


def list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# context history-[("user", 'msg'),...]
# persona_texts ['fact1', ...]
# persona_vecs [tensor, ...]
# persona_ranks (all:[int, ...], relevant:[int, ...])
# persona_ranks_approved True/False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["history"] = []
    context.chat_data["persona_texts"] = []
    context.chat_data["persona_vecs"] = []
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="все начинается с чистого листа!",
    )


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="""
        Это диалоговая модель, обученная на TolokaPersonaChat.
        Репозиторий модели: https://github.com/Anpopaicoconat/persona
        Для начала диалога используйте /start.
        Для создания персоны бота, введите /persona и указывайте факты с новой строки.
        Для сброса всего контекста и начала нового диалога используйте /start.
        """,
    )


DEFOLT_PERSONA = [
    "я Саша",
    "Я работаю инженером.",
    "У меня трое детей",
    "У меня есть котенок",
    "Я живу в городе Москва",
    "Я люблю рисовать",
    "Имею высшее образование",
    "У меня своя машина",
    "Хобби - рыбалка",
    "Люблю слушать шансон",
    "У меня большая семья.",
    "Мне нравится лето.",
    "Я люблю комедии.",
    "Я знаю четыре языка.",
    "У меня есть дача.",
    "Я люблю цитрусовые.",
    "Я люблю читать книги.",
    "Я мечтаю о море",
    "Я люблю детей",
    "Я никогда не видела жирафа",
]


async def add_persona(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # proc persona
    texts = update.message.text.split("\n")[1:]
    if len(texts) < 1:
        texts = DEFOLT_PERSONA
    vecs = model.calculate_candidats(texts)
    context.chat_data["persona_texts"] = context.chat_data["persona_texts"] + texts
    context.chat_data["persona_vecs"] = (
        context.chat_data["persona_vecs"] + vecs.tolist()
    )

    # reply
    confirm_msg = "персона обновленна!\n" + "\n".join(
        context.chat_data["persona_texts"]
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=confirm_msg,
    )


# async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     context.chat_data["history"].append(("user", update.message.text))
#     relevant_gk, all_gks = model.retrieve_gk(
#         context.chat_data["history"],
#         context.chat_data["persona_texts"],
#         context.chat_data["persona_vecs"],
#     )
#     msg, new_gks = model.generate_reply(context.chat_data["history"], relevant_gk)
#     context.chat_data["history"].append(("model", msg))

#     all_gks = ["{}|{}".format(round(t[0], 2), t[1]) for t in all_gks]
#     relevant_gk = ["{}|{}".format(round(t[0], 2), t[1]) for t in relevant_gk]
#     all_gks = "все ранги:\n" + "\n".join(all_gks)
#     relevant_gk = "релевантные кандидаты:\n" + "\n".join([str(t) for t in relevant_gk])
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=all_gks)
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=relevant_gk)
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


async def rank_cand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["history"].append(("user", update.message.text))

    relevant_gk_idx, distances = model.retrieve_gk(
        context.chat_data["history"],
        context.chat_data["persona_vecs"],
    )
    context.chat_data["persona_ranks"] = relevant_gk_idx
    # reply
    all_gks = [
        "{}|{}|".format(t[0], t[1][0], round(t[1][1], 2))
        for t in enumerate(zip(context.chat_data["persona_texts"], distances))
    ]
    relevant_gk = [
        (i, context.chat_data["persona_texts"][i], distances[i])
        for i in relevant_gk_idx
    ]
    relevant_gk = ["{}|{}|{}".format(t[0], t[1], round(t[2], 2)) for t in relevant_gk]

    all_gks = "все кандидаты:\n" + "\n".join(all_gks)
    relevant_gk = "релевантные кандидаты:\n" + "\n".join([str(t) for t in relevant_gk])
    end_quest = """если выбраны верные знания о персоне введите /cand ok
    если нет, введите /cand 1 n ... , где n номер ранга"""
    await context.bot.send_message(chat_id=update.effective_chat.id, text=all_gks)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=relevant_gk)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=end_quest)


async def rerank_cand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ranks_idxs = update.message.text.split(" ")[1:]

    if list_get(ranks_idxs, 0, 0) != "ok":
        context.chat_data["persona_ranks"] = ranks_idxs

    relevant_gk = [
        context.chat_data["persona_texts"][int(i)]
        for i in context.chat_data["persona_ranks"]
    ]
    msg, new_gks = model.generate_reply(context.chat_data["history"], relevant_gk)
    context.chat_data["history"].append(("model", msg))
    context.chat_data["persona_texts"].append(new_gks)
    new_gks = "\n".join(new_gks)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="новые знания о персоне:\n" + new_gks
    )


if __name__ == "__main__":
    # proxy
    os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"

    # model
    bi_encoder = BERT_RetrievalModel.load_from_checkpoint(
        "/home/stc/persona/logs/bi_encoder/36037371cee4404b80aa618268a2e24c/checkpoints/epoch=29-step=22080.ckpt"
    )
    bi_encoder.eval()
    generative = GPT_GenerativeModel.load_from_checkpoint(
        "/home/stc/persona/logs/gpt-epoch=00-val_loss=3.62.ckpt"
    )
    generative.eval()
    model = BiEncoder_GPT(
        retrieval_model=bi_encoder,
        generative_model=generative,
    )

    # bot
    application = (
        Application.builder()
        .token("5330579133:AAGfO3lGtzLP-xKd3uJw_myqaqTfV6Hw1ac")
        .build()
    )

    start_handler = CommandHandler("start", start)
    add_persona_handler = CommandHandler("persona", add_persona)
    # reply_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), reply)
    rank_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), rank_cand)
    rerank_handler = CommandHandler("cand", rerank_cand)

    application.add_handler(start_handler)
    application.add_handler(add_persona_handler)
    application.add_handler(rank_handler)
    application.add_handler(rerank_handler)

    application.run_polling()
