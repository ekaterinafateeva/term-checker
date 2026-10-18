import pandas as pd
import streamlit as st

from document_check import DEFAULT_THRESHOLD, TermChecker, highlight_html
from functions import load_glossary
from glossary_check import DUP_THRESHOLD, find_duplicates, quality_issues, similarity_matrix

st.set_page_config(page_title="Проверка терминологии", layout="wide")


# Streamlit перезапускает весь файл при каждом нажатии кнопки.
# cache_resource загружает справочник и модель один раз и держит их в памяти.
@st.cache_resource
def get_checker():
    return TermChecker(load_glossary())


@st.cache_resource
def get_audit_data():
    df = load_glossary()
    return df, quality_issues(df), similarity_matrix(df)


st.title("Проверка документов на соответствие справочнику терминов")
tab_check, tab_audit = st.tabs(["Проверка документа", "Аудит справочника"])

with tab_check:
    # Текст документа: загруженный файл или, по умолчанию, первая записка
    uploaded = st.file_uploader("Загрузите документ .txt или вставьте текст ниже", type="txt")
    if uploaded:
        text = uploaded.read().decode("utf-8")
    else:
        with open("data/sample_doc.txt", encoding="utf-8") as f:
            text = f.read()
    text = st.text_area("Текст документа", text, height=260)
    threshold = st.slider("Порог близости по смыслу", 0.30, 0.95, DEFAULT_THRESHOLD, 0.05,
                          help="Ниже — больше находок и больше ложных замечаний")

    # По кнопке: проверить текст, показать его с подсветкой и таблицу замечаний
    if st.button("Проверить", type="primary"):
        with st.spinner("Проверяю..."):
            findings = get_checker().check(text, threshold)
        st.markdown(f"**Замечаний: {len(findings)}.** "
                    "Красным — выражение записано в справочнике как неправильный вариант термина. "
                    "Жёлтым — выражения нет в справочнике, но по смыслу оно близко к термину.")
        st.markdown(f'<div style="line-height:1.9">{highlight_html(text, findings)}</div>',
                    unsafe_allow_html=True)
        if findings:
            table = pd.DataFrame(findings)[["phrase", "suggestion", "method", "score", "sentence"]]
            table.columns = ["Фраза", "Используйте", "Метод", "Уверенность", "Предложение"]
            st.dataframe(table, use_container_width=True, hide_index=True)

with tab_audit:
    # Аудит справочника: проблемы качества и дубли с настраиваемым порогом
    df, issues, sim = get_audit_data()
    dup_threshold = st.slider("Порог близости для дублей по смыслу", 0.20, 0.80, DUP_THRESHOLD, 0.05)
    dups = find_duplicates(df, sim, dup_threshold)

    st.subheader(f"Проблемы качества: {len(issues)}")
    st.dataframe(issues, use_container_width=True, hide_index=True)

    st.subheader(f"Возможные дубли: {len(dups)}")
    st.caption("Дубли по названию надёжны. Дубли по смыслу — кандидаты: среди них бывают "
               "смежные понятия, окончательное решение принимает эксперт.")
    st.dataframe(dups, use_container_width=True, hide_index=True)