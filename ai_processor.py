import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from database import Database

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self, db):
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.site_url = os.getenv('SITE_URL', 'https://async-news.ru')
        self.site_name = os.getenv('SITE_NAME', 'AsyncNews')
        self.db = db
        
        # НОВЫЙ ПРОМПТ с @infinewss в конце
        self.prompt_template = """
Ты – профессиональный новостной редактор. Твоя задача – переработать предоставленный новостной текст для публикации в Telegram-канале.

Требования:
- Сделай текст кратким (не более 500 символов).
- Сохрани все ключевые факты: что произошло, где, когда, кто участники.
- Убери воду, клише, повторы.
- Переформулируй, чтобы текст был живым и понятным широкой аудитории.
- В конце добавь хэштеги (не более 3), отражающие тему (например, #Технологии #Rust #Новости).
- Оригинал пиши на русском языке (если новость на другом языке – переведи на русский и отредактируй).
- После хэштегов добавь "@infinewss".

Исходный новостной текст:
---
{original_content}
---
Твой ответ должен содержать только отредактированный текст, хэштеги и "@infinewss" – без лишних пояснений.
        """

        # Инициализация клиента OpenAI с OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key
        )

    def process_news(self, news_item):
        """Обработка новости с помощью AI"""
        try:
            content = news_item.get('content', '')
            content_length = len(content)
            
            if content_length < 50:
                logger.warning(f"Контент новости '{news_item['title']}' слишком короткий ({content_length} символов). Качество обработки может быть низким.")
            else:
                logger.info(f"Обработка новости '{news_item['title']}' с контентом длиной {content_length} символов")
            
            prompt = self.prompt_template.format(
                original_content=content
            )

            logger.info(f"Отправка новости '{news_item['title']}' на обработку AI через OpenRouter")

            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                extra_body={},
                model="google/gemini-2.0-flash-lite-001",  # бесплатная модель
                messages=[
                    {"role": "system", "content": "Ты - редактор новостей для Telegram-канала."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            ai_response = completion.choices[0].message.content.strip()

            # Если ответ пустой — логируем ошибку
            if not ai_response:
                logger.error(f"AI вернул пустой ответ для новости '{news_item['title']}'")
                return {"success": False, "error": "Empty AI response"}

            # Разделение на заголовок и контент (если нужно)
            lines = ai_response.split('\n')
            processed_title = lines[0] if lines else ""
            processed_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

            # Сохранение обработанной новости в базу данных
            processed_id = self.db.save_processed_news(
                news_item['id'],
                processed_title,
                processed_content
            )

            if processed_id:
                logger.info(f"Новость успешно обработана и сохранена с ID {processed_id}")
                return {
                    "id": news_item['id'],
                    "processed_title": processed_title,
                    "processed_content": processed_content,
                    "success": True
                }
            else:
                logger.error(f"Не удалось сохранить обработанную новость в базу данных")
                return {"success": False, "error": "Database error"}

        except Exception as e:
            logger.error(f"Ошибка при обработке новости: {e}")
            return {"success": False, "error": str(e)}

    def process_batch(self, news_items, batch_size=5):
        """Обработка пакета новостей с учетом качества контента"""
        results = []
        count = 0
        skipped = 0

        for news_item in news_items:
            if count >= batch_size:
                break

            content_length = len(news_item.get('content', ''))
            if content_length < 50:
                logger.warning(f"Пропуск новости '{news_item['title']}' из-за недостаточного контента ({content_length} символов)")
                results.append({
                    "id": news_item['id'],
                    "success": False,
                    "error": "Insufficient content length",
                    "content_length": content_length
                })
                skipped += 1
                continue

            result = self.process_news(news_item)
            results.append(result)

            if result["success"]:
                count += 1
            else:
                logger.error(f"Ошибка при обработке новости '{news_item['title']}': {result.get('error', 'Неизвестная ошибка')}")

        logger.info(f"Обработано {count} новостей, пропущено {skipped} из {len(news_items)}")
        return results
