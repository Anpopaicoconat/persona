import os
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId

sys.path.append("/home/stc/persona/")
from models import BERT_RetrievalModel, GPT_GenerativeModel
from inference_pipeline import BiEncoder_GPT
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    filters,
    Application,
    ApplicationBuilder,
    MessageHandler,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["mod"] = "smpl"
    context.chat_data["history"] = []
    context.chat_data["persona_texts"] = []
    context.chat_data["persona_vecs"] = []
    context.chat_data["persona_ranks_all"] = []
    context.chat_data["persona_ranks"] = []
    context.chat_data["metric"] = {"logic": 0, "spec": 0, "person": 0}
    context.chat_data["dialog_id"] = None
    context.chat_data["need_answer"] = False
    context.chat_data["need_metric"] = False
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Персона бота и история сообщений очишена. Диалог начинается сначала.",
    )


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_html = ""
    await update.message.reply_html(text=help_html)


async def set_smpl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["mod"] = "smpl"
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Включен режим простого диалога",
    )


async def set_val(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["mod"] = "val"
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Включен режим валидации и разметки",
    )


async def add_persona(update: Update, context: ContextTypes.DEFAULT_TYPE):
    defolt_persona = [
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
    # proc persona
    texts = update.message.text.split("\n")[1:]
    if len(texts) < 1:
        texts = defolt_persona
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


async def msg_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["history"].append(("user", update.message.text))

    if context.chat_data["mod"] == "smpl":
        msg, new_gks = generate(update, context)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
        if new_gks:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Новые знания о персоне:\n" + new_gks,
            )
    elif context.chat_data["mod"] == "val":
        context.chat_data["need_answer"] = True
        context.chat_data["need_metric"] = True
        if len(context.chat_data["persona_texts"]) == 0:
            msg, new_gks = generate(update, context)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
            if new_gks:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Новые знания о персоне:\n" + new_gks,
                )
        else:
            relevant_gk_idx, distances = model.retrieve_gk(
                context.chat_data["history"],
                context.chat_data["persona_vecs"],
                top_k=10,
                th=-1,
            )
            context.chat_data["persona_ranks_all"] = relevant_gk_idx
            context.chat_data["persona_ranks"] = []
            # даем кандидатов на выбор
            msg_text, reply_markup = rank_buttons(
                context.chat_data["persona_texts"],
                context.chat_data["persona_ranks_all"],
                context.chat_data["persona_ranks"],
            )
            await update.message.reply_text(msg_text, reply_markup=reply_markup)


def rank_buttons(persona_texts, persona_ranks_all, persona_ranks):
    comand = "rank_"
    keyboard = [
        [InlineKeyboardButton(persona_texts[i], callback_data=comand + str(i))]
        for i in persona_ranks_all
    ]
    keyboard.append([InlineKeyboardButton("готово", callback_data=comand + "generate")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    if len(persona_ranks) == 0:
        msg_text = "Выберите подходящие факты:"
    else:
        msg_text = "\n".join(persona_texts[i] for i in persona_ranks)
        msg_text = f"Выбранные факты:\n{msg_text}"
    return msg_text, reply_markup


def metric_buttons(metrics):
    comand = "metric_"
    labels = []
    for text, marker, metric in zip(
        ["логичен", "специфичен", "персонален"], ["👌", "👍", "❤️"], metrics
    ):
        if metrics[metric]:
            labels.append(text + marker)
        else:
            labels.append(text)
    keyboard = [
        [
            InlineKeyboardButton(text, callback_data=comand + data)
            for text, data in zip(labels, metrics)
        ]
    ]
    keyboard.append([InlineKeyboardButton("готово", callback_data=comand + "save")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    msg_text = "Оцените ответ."
    return msg_text, reply_markup


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if "rank_" in query.data:
        if context.chat_data["need_answer"]:
            comand = query.data.split("_")[-1]
            if comand != "generate":
                comand = int(comand)
                if comand in context.chat_data["persona_ranks"]:
                    context.chat_data["persona_ranks"].remove(comand)
                else:
                    context.chat_data["persona_ranks"].append(comand)

                msg_text, reply_markup = rank_buttons(
                    context.chat_data["persona_texts"],
                    context.chat_data["persona_ranks_all"],
                    context.chat_data["persona_ranks"],
                )
                await query.edit_message_text(text=msg_text, reply_markup=reply_markup)
            else:
                # посылаем сообщение с ответом
                msg, new_gks = generate(update, context)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=msg
                )
                if new_gks:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="Новые знания о персоне:\n" + new_gks,
                    )
                # создаем сообщение с метриками
                context.chat_data["metric"] = {"logic": 0, "spec": 0, "person": 0}
                msg_text, reply_markup = metric_buttons(context.chat_data["metric"])
                # await update.message.reply_text(msg_text, reply_markup=reply_markup)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=msg_text,
                    reply_markup=reply_markup,
                )

    elif "metric_" in query.data:
        if context.chat_data["need_metric"]:
            comand = query.data.split("_")[-1]
            if comand == "save":
                save_masg(update, context)
            else:
                context.chat_data["metric"][comand] = (
                    context.chat_data["metric"][comand] - 1
                ) ** 2
                msg_text, reply_markup = metric_buttons(context.chat_data["metric"])
                await query.edit_message_text(text=msg_text, reply_markup=reply_markup)


def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["need_answer"] = False
    context.chat_data["need_metric"] = True
    relevant_gk = [
        context.chat_data["persona_texts"][int(i)]
        for i in context.chat_data["persona_ranks"]
    ]
    msg, new_gks = model.generate_reply(context.chat_data["history"], relevant_gk)
    context.chat_data["history"].append(("model", msg))
    # генерация знаний
    if new_gks:
        vecs = model.calculate_candidats(new_gks)
        context.chat_data["persona_texts"] = (
            context.chat_data["persona_texts"] + new_gks
        )
        context.chat_data["persona_vecs"] = (
            context.chat_data["persona_vecs"] + vecs.tolist()
        )
        new_gks = "\n".join(new_gks)

    return msg, new_gks


def save_masg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["need_answer"] = False
    context.chat_data["need_metric"] = False
    persons = [[], context.chat_data["persona_texts"]]
    user_msg = {
        "person": 0,
        "text": context.chat_data["history"][-2][1],
        "gk": [],
        "metric": context.chat_data["metric"],
    }
    model_msg = {
        "person": 1,
        "text": context.chat_data["history"][-1][1],
        "gk": context.chat_data["persona_ranks"],
        "metric": context.chat_data["metric"],
    }
    if context.chat_data["dialog_id"] is None:
        masages = [user_msg, model_msg]
        data = {
            "user_id": update.effective_chat.id,
            "persons": persons,
            "dialog": masages,
        }
        dialog_id = collection.insert_one(data).inserted_id
        context.chat_data["dialog_id"] = dialog_id
    else:
        dialog_data = collection.find_one({"_id": context.chat_data["dialog_id"]})
        dialog = dialog_data["dialog"] + [user_msg, model_msg]
        collection.replace_one(
            {"_id": dialog_data["_id"]},
            {
                "user_id": update.effective_chat.id,
                "persons": persons,
                "dialog": dialog,
            },
        ).upserted_id
    # reset metrics
    context.chat_data["metric"] = {"logic": 0, "spec": 0, "person": 0}


if __name__ == "__main__":
    # proxy
    print("proxy")
    os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"
    # mongodb
    print("mongodb")
    client = MongoClient("localhost", 27017)
    db = client["ChatBot-Data"]
    collection = db["dialogs"]
    # model
    print("model")
    bi_encoder = BERT_RetrievalModel.load_from_checkpoint(
        "/home/stc/persona/logs/bi_encoder/36037371cee4404b80aa618268a2e24c/checkpoints/epoch=29-step=22080.ckpt"
    )
    bi_encoder.eval()
    generative = GPT_GenerativeModel.load_from_checkpoint(
        "/home/stc/persona/logs/gpt_answer/gpt-epoch=00-val_loss=3.62.ckpt"
    )
    generative.eval()
    model = BiEncoder_GPT(
        retrieval_model=bi_encoder,
        generative_model=generative,
    )
    # bot
    print("bot")
    application = Application.builder().token("").build()
    print("handler")
    reply_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), msg_handler)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("set_simple", set_smpl))
    application.add_handler(CommandHandler("set_validation", set_val))
    application.add_handler(CommandHandler("persona", add_persona))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(reply_handler)
    print("run_polling")
    application.run_polling()
