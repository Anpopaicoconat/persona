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

# mongodb
def insert_document(
    collection, dialog_id, user_id, history, persona_ranks, metric, persona_texts
):
    dialog_id = ObjectId(dialog_id)
    dialog_data = collection.find_one({"_id": dialog_id})

    user_msg = {"person": 0, "text": history[-2][1], "gk": [], "metric": metric}
    model_msg = {
        "person": 1,
        "text": history[-1][1],
        "gk": persona_ranks,
        "metric": metric,
    }
    persons = [[], persona_texts]

    if dialog_data is None:
        masages = [user_msg, model_msg]
        data = {"user_id": user_id, "persons": persons, "dialog": masages}
        dialog_id = collection.insert_one(data).inserted_id
    else:
        dialog_data["dialog"].append(user_msg)
        dialog_data["dialog"].append(model_msg)
        collection.replace_one(
            {"_id": dialog_data["_id"]},
            {"user_id": user_id, "persons": persons, "dialog": dialog_data["dialog"]},
        ).upserted_id

    return str(dialog_id)


#############################################


def list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# mod 'val'/'smpl'
# context history [("user", 'msg'),...]
# persona_texts ['fact1', ...]
# persona_vecs [tensor, ...]
# persona_ranks (all:[int, ...], relevant:[int, ...])
# persona_ranks_all (all:[int, ...], relevant:[int, ...]) > persona_ranks


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["mod"] = "val"
    context.chat_data["history"] = []
    context.chat_data["persona_texts"] = []
    context.chat_data["persona_vecs"] = []
    set_smpl(update, context)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="все начинается с чистого листа!",
    )


help_html = """
<b>Это персонофицированная гибридная диалоговая модель, обученная на TolokaPersonaChat.</b>
Для генерации ответов такие модели используют знания о своей персоне. Ранжируя кандидатов она отбирает релевантные знания о себе и затем использует их для последующей генерации ответа.
<b>Репозиторий модели:</b> https://github.com/Anpopaicoconat/persona

<b>Список команд:</b>
- Для начала диалога или что бы закончить прошлый используйте /start.
- Для создания персоны бота по умолчанию, используйте /persona 
- Что бы добавить новые факты в персону модели используйте /persona и указывайте факты с новой строки. (что бы создать полностью собственную персону, указывайте факты изначально, не устанавливая дефолтную персону)
Бот может работать в двух режимах. 
- Для простой беседы с ботом используйте /set_simple в этом режиме бот отвечает на ваши сообщения и дополнительно может указать на какие знания о себе он опирался генерируя ответ, а так же может сгенерировать новое знание о себе, об этом он так же дополнительно сообщит вам.
- Для беседы с расширенными возможностями управления используйте /set_validation в этом режиме бот в ответ на вашу реплику сначала проранжирует знания о себе и предоставит вам на выбор 10 наиболее релевантных фактов. После этого вы можете выбрать наиболее подходящие из них. Выбранный вами список будет использован для генерации ответа модели. После вам будет предложено оценить ответ. Ваша реплика, выбранные вами факты и оценка ответа модели будут сохранены и использованны для улучшения модели.

<b>быстрый старт:</b>
/start
/set_simple
/persona

<b>быстрый старт в ручном режиме:</b>
/start
/set_validation
/persona

P.S. генерация ответа требует времени (в среднем 30 сек), дождитесь ответа и только после этого отправляейте следующее сообщение.
"""


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
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


def smpl_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.chat_data["history"].append(("user", update.message.text))
    relevant_gk_idx, distances = model.retrieve_gk(
        context.chat_data["history"],
        context.chat_data["persona_vecs"],
        top_k=3,
        th=0,
    )
    relevant_gk = [context.chat_data["persona_texts"][int(i)] for i in relevant_gk_idx]
    msg, new_gks = model.generate_reply(context.chat_data["history"], relevant_gk)
    context.chat_data["history"].append(("model", msg))
    if len(relevant_gk) > 0:
        relevant_gk = "Ответ был сделан с опорой на следующие знания:\n" + "\n".join(
            [str(t) for t in relevant_gk]
        )
    else:
        relevant_gk = None
    if len(new_gks) > 0:
        new_gks = "В ходе ответа я понял о себе следующиее:\n" + "\n".join(
            [str(t) for t in new_gks]
        )
    else:
        new_gks = None
    return relevant_gk, msg, new_gks


def save_answers(update, context):
    metric = context.chat_data.get("metric", False)
    if metric:
        dialog_id = context.chat_data.get("dialog_id", None)
        id = insert_document(
            collection=DIALOG_TABLE,
            dialog_id=dialog_id,
            user_id=update.effective_chat.id,
            history=context.chat_data["history"],
            persona_ranks=context.chat_data["persona_ranks"],
            metric=metric,
            persona_texts=context.chat_data["persona_texts"],
        )
        return id


async def msg_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.chat_data["mod"] == "smpl":
        relevant_gk, msg, new_gks = smpl_reply(update, context)
        if relevant_gk is not None:
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=relevant_gk
            )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
        if new_gks is not None:
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=new_gks
            )
    elif context.chat_data["mod"] == "val":
        # сохраняем предыдущий ответ
        dialog_id = save_answers(update, context)
        context.chat_data["dialog_id"] = dialog_id
        context.chat_data["history"].append(("user", update.message.text))

        # retrieve
        relevant_gk_idx, distances = model.retrieve_gk(
            context.chat_data["history"],
            context.chat_data["persona_vecs"],
            top_k=10,
            th=-1,
        )
        context.chat_data["persona_ranks_all"] = relevant_gk_idx
        context.chat_data["persona_ranks"] = []
        context.chat_data["metric"] = {"logic": 0, "spec": 0, "person": 0}
        # reply
        keyboard = [
            [
                InlineKeyboardButton(
                    context.chat_data["persona_texts"][i], callback_data=i
                )
            ]
            for i in context.chat_data["persona_ranks_all"]
        ]
        keyboard.append([InlineKeyboardButton("готово", callback_data="generate")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "выберите подходящие факты:", reply_markup=reply_markup
        )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query
    if query.data.isdigit():
        context.chat_data["persona_ranks"].append(int(query.data))
        keyboard = [
            [
                InlineKeyboardButton(
                    context.chat_data["persona_texts"][i], callback_data=i
                )
            ]
            for i in context.chat_data["persona_ranks_all"]
        ]
        keyboard.append([InlineKeyboardButton("готово", callback_data="generate")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.answer()
        gks = "\n".join(
            context.chat_data["persona_texts"][i]
            for i in context.chat_data["persona_ranks"]
        )
        await query.edit_message_text(
            text=f"Выбранные факты:\n{gks}", reply_markup=reply_markup
        )
    elif query.data == "generate":
        relevant_gk = [
            context.chat_data["persona_texts"][int(i)]
            for i in context.chat_data["persona_ranks"]
        ]
        msg, new_gks = model.generate_reply(context.chat_data["history"], relevant_gk)
        context.chat_data["history"].append(("model", msg))
        context.chat_data["persona_texts"] += new_gks
        new_gks = "\n".join(new_gks)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
        if new_gks:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Новые знания о персоне:\n" + new_gks,
            )

        keyboard = [
            [
                InlineKeyboardButton("логичен 👌", callback_data="logic"),
                InlineKeyboardButton("специфичен 👍", callback_data="spec"),
                InlineKeyboardButton("персонален ❤️", callback_data="person"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        metric = context.chat_data.get("metric", False)
        answer = ", ".join(
            [
                x
                for x, y in zip(["логичен", "специфичен", "персонален"], metric)
                if metric[y]
            ]
        )
        gks = "\n".join(
            context.chat_data["persona_texts"][i]
            for i in context.chat_data["persona_ranks"]
        )
        await query.edit_message_text(
            text=f"Выбранные факты:\n{gks}\nОтвет был: {answer}",
            reply_markup=reply_markup,
        )
    else:
        context.chat_data["metric"][query.data] = 1
        keyboard = [
            [
                InlineKeyboardButton("логичен 👌", callback_data="logic"),
                InlineKeyboardButton("специфичен 👍", callback_data="spec"),
                InlineKeyboardButton("персонален ❤️", callback_data="person"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        metric = context.chat_data.get("metric", False)
        answer = ", ".join(
            [
                x
                for x, y in zip(["логичен", "специфичен", "персонален"], metric)
                if metric[y]
            ]
        )
        gks = "\n".join(
            context.chat_data["persona_texts"][i]
            for i in context.chat_data["persona_ranks"]
        )
        await query.edit_message_text(
            text=f"Выбранные факты:\n{gks}\nОтвет был: {answer}",
            reply_markup=reply_markup,
        )


if __name__ == "__main__":
    # proxy
    os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
    os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"
    # bongodb
    client = MongoClient("localhost", 27017)
    db = client["ChatBot-Data"]
    DIALOG_TABLE = db["dialogs"]
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
        .token("5330579133:AAHLN46Kqp-Vl8Gz8j-dvHbRpKL_NwtjKQ4")
        .build()
    )

    rank_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), msg_handler)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("set_simple", set_smpl))
    application.add_handler(CommandHandler("set_validation", set_val))
    application.add_handler(CommandHandler("persona", add_persona))
    application.add_handler(CallbackQueryHandler(button))
    # application.add_handler(reply_handler)
    application.add_handler(rank_handler)

    application.run_polling()
