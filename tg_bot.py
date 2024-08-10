import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
model = None
model = tf.keras.models.load_model('ModelDist_36.keras')
classes = ['NORMAL', 'DRUSEN', 'DME', 'CNV']


def predict(im):
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    im = tf.math.reduce_mean(im, axis=2, keepdims=True)
    im = tf.image.resize_with_pad(im, target_height=496, target_width=512)
    im = im.numpy().astype(np.ubyte)
    pred = np.argmax(model(im))
    return classes[pred]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(rf"Hi {user.mention_html()}!")
    await update.message.reply_text("You can send here an optical coherence tomography image or path to such an image "
                                    "or path to a folder with images. The bot will give you a prediction "
                                    "of the image class (NORMAL, CNV, DME or DRUSEN).")


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await update.message.reply_text("You can send here an optical coherence tomography image or path to such an image "
                                    "or path to a folder with images. The bot will give you a prediction "
                                    "of the image class (NORMAL, CNV, DME or DRUSEN).")


async def classify_by_path(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = update.message.text
    if os.path.isfile(path):
        print(os.path.isfile(path))
        try:
            im = cv2.imread(path)
            res = predict(im)
        except:
            res = 'Invalid Image Format'
        await update.message.reply_text(res)
    elif os.path.isdir(path):
        for im_path in os.listdir(path):
            try:
                im = cv2.imread(path + '/' + im_path)
                res = predict(im)
            except:
                res = 'Invalid Image Format'
            await update.message.reply_text(res)
    else:
        res = 'Invalid Path'
        await update.message.reply_text(res)


async def downloader(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Download file
    new_file = await update.message.effective_attachment[-1].get_file()
    file = await new_file.download_to_drive()
    return file


async def classify_by_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
            not update.message
            or not update.effective_chat
            or (
            not update.message.photo
            and not update.message.video
            and not update.message.document
            and not update.message.sticker
            and not update.message.animation
    )
    ):
        return
    file = await downloader(update, context)

    if not file:
        await update.message.reply_text("Something went wrong, try again")
        return
    try:
        image = Image.open(file)
        im = np.array(image)
        res = predict(im)
    except:
        res = 'Invalid Image Format'
    file.unlink()
    await update.message.reply_text(res)


def main():
    application = Application.builder().token("Token").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_by_path))

    application.add_handler(MessageHandler(filters.PHOTO, classify_by_file))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
