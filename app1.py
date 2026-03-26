import streamlit as st
import streamlit1
import sherbakova
import app
import modeltextimage
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity

st.title("Многостраничное Streamlit-приложение")
page = st.sidebar.selectbox("Выберите страницу", ["Резюмирование текста","Белоглазова", "Усачёва", "Щербакова. Титаник", "Белоглазова. Нейронка", "Щербакова. Нейронка"])
if page == "Белоглазова":
    streamlit1.show_page()
elif page == "Усачёва":
    app.show_page()
elif page == "Щербакова. Титаник":
    sherbakova.show_page()
elif page == "Резюмирование текста":
    modeltextimage.show_page()
elif page == "Белоглазова. Нейронка":
    st.title("Кросс-лингвистическое сравнение текстов (mmBERT)")
    
    # Загрузка модели и токенизатора (один раз через Streamlit кэш)
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        model = AutoModel.from_pretrained("jhu-clsp/mmBERT-base")
        return tokenizer, model
    
    tokenizer, model = load_model()
    def get_embeddings(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Усреднение по последнему скрытому состоянию (по токенам)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
    st.write("Введите несколько предложений на разных языках для сравнения:")
    input_texts = st.text_area(
        "Тексты (по одному на строку)",
        value="""Artificial intelligence is transforming technology\nLa inteligencia artificial está transformando la tecnología\nL'intelligence artificielle transforme la technologie\n人工智能正在改变技术""",
        height=150
    )
    
    show_matrix = st.checkbox("Показать полную матрицу схожести", True)
    
    if st.button("Вычислить схожесть"):
        if not input_texts.strip():
            st.warning("Пожалуйста, введите хотя бы одно предложение.")
        else:
            texts = [line.strip() for line in input_texts.split("\n") if line.strip()]
            
            if len(texts) < 2:
                st.warning("Введите хотя бы два предложения для сравнения.")
            else:
                with st.spinner("Обработка текстов..."):
                    try:
                        embeddings = get_embeddings(texts)
                        similarities = cosine_similarity(embeddings)
    
                        st.success("Сравнение завершено!")
    
                        # Отображение результатов
                        st.write("### Матрица косинусной схожести:")
                        if show_matrix:
                            st.dataframe(similarities.round(4))
                        else:
                            st.write("Показаны только схожести с первым предложением:")
                            first_similarities = similarities[0].round(4)
                            sim_table = {
                                "Текст": texts,
                                "Схожесть с первым": first_similarities
                            }
                            st.table(sim_table)
    
                        # Опционально: визуализация тепловой карты
                        if st.checkbox("Показать тепловую карту"):
                            import seaborn as sns
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(similarities, annot=True, xticklabels=[t[:10] + "..." for t in texts],
                                        yticklabels=[t[:10] + "..." for t in texts], cmap='Blues', ax=ax)
                            plt.title("Косинусная схожесть (эмбеддинги mmBERT)")
                            st.pyplot(fig)
    
                    except Exception as e:
                        st.error(f"Ошибка при обработке: {str(e)}")
    
    # Информация о модели
    with st.expander("ℹ️ О модели mmBERT"):
        st.markdown("""
        Модель **jhu-clsp/mmBERT-base** — это мультиязычная BERT-модель, обученная на параллельных корпусах для улучшения кросс-лингвистических представлений.
        Она позволяет сравнивать семантическое сходство текстов на разных языках.
        """)
elif page == "Щербакова. Нейронка":
    st.title("Проверка текста на спам")

    @st.cache_resource
    def load_spam_model():
        try:
            with st.spinner("Загружаю модель RUSpam/spam_deberta_v4... Пожалуйста, подождите."):
                model = pipeline("text-classification", model="RUSpam/spam_deberta_v4")
            return model
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {e}")
            st.stop()

    textclassification = load_spam_model()
    st.success("Модель успешно загружена и готова к работе!")

    text = st.text_area("Введите текст для проверки:", "Вы выиграли приз! Переходите по ссылке для получения")

    if st.button("Проверить"):
        if not text.strip():
            st.warning("Введите текст для проверки")
        else:
            with st.spinner("🔍 Анализирую текст..."):
                result = textclassification(text)[0]
                label = result["label"]
                score = result["score"]
                verdict = "Спам" if label == "LABEL_1" else "Не спам"

            st.subheader(f"Результат: **{verdict}**")
            st.progress(score)
            st.write(f"Уверенность модели: {score*100:.2f}%")
