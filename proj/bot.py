import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
from pathlib import Path

# Определяем состояния для ConversationHandler
PLATES, FIO, ROOM, PHONE = range(4)

# Путь к JSON-файлу
DATA_FILE = Path("car_data.json")

# Загрузка данных из JSON-файла
def load_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Сохранение данных в JSON-файл
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Команда /start
async def start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text(
        "Введите номер автомобиля (используя лат. символы):"
    )
    return PLATES

# Обработка номера автомобиля
async def get_plates(update: Update, context: CallbackContext) -> int:
    context.user_data['plates'] = update.message.text
    await update.message.reply_text("Введите ФИО:")
    return FIO

# Обработка ФИО
async def get_fio(update: Update, context: CallbackContext) -> int:
    context.user_data['fio'] = update.message.text
    await update.message.reply_text("Введите номер комнаты:")
    return ROOM

# Обработка номера комнаты
async def get_room(update: Update, context: CallbackContext) -> int:
    context.user_data['room'] = update.message.text
    await update.message.reply_text("Введите номер телефона:")
    return PHONE

# Обработка номера телефона и сохранение данных в JSON
async def get_phone(update: Update, context: CallbackContext) -> int:
    context.user_data['phone'] = update.message.text

    # Формируем данные для сохранения
    new_car = {
        "plates": context.user_data['plates'],
        "fio": context.user_data['fio'],
        "room": context.user_data['room'],
        "phone": context.user_data['phone']
    }

    # Загружаем существующие данные
    data = load_data()
    data.append(new_car)  # Добавляем новую запись
    save_data(data)  # Сохраняем обновленные данные

    await update.message.reply_text("Заявка отправлена!")
    return ConversationHandler.END

# Команда /cancel для отмены диалога
async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Добавление автомобиля отменено.")
    return ConversationHandler.END

def main() -> None:
    # Вставьте сюда ваш токен
    application = Application.builder().token("token").build()

    # Определяем ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            PLATES: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_plates)],
            FIO: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_fio)],
            ROOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_room)],
            PHONE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_phone)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()