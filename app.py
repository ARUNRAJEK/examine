import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# Load API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("LLM Based Exam Answer Evaluator")

# Upload files
blueprint_file = st.file_uploader("Upload Blueprint CSV", type=["csv"])
student_file = st.file_uploader("Upload Student Answer CSV", type=["csv"])


if blueprint_file and student_file:

    blueprint_df = pd.read_csv(blueprint_file)
    student_df = pd.read_csv(student_file)

    merged_df = pd.merge(blueprint_df, student_df, on="Question")

    results = []

    if st.button("Evaluate Answers"):

        with st.spinner("Evaluating using LLM..."):

            for _, row in merged_df.iterrows():

                question = row["Question"]
                teacher_answer = row["Answer"]
                student_answer = row["Student_Answer"]
                max_marks = row["Marks"]

                prompt = f"""
                You are an exam evaluator.

                Question: {question}

                Teacher Answer: {teacher_answer}

                Student Answer: {student_answer}

                Maximum Marks: {max_marks}

                Give marks based on correctness and similarity.
                Only return numeric marks.
                """

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )

                marks = response.choices[0].message.content.strip()

                try:
                    marks = float(marks)
                except:
                    marks = 0

                results.append(marks)

        merged_df["Obtained Marks"] = results

        # Total score
        total = merged_df["Marks"].sum()
        obtained = merged_df["Obtained Marks"].sum()

        st.subheader("Evaluation Result")
        st.dataframe(merged_df)

        st.success(f"Total Score: {obtained} / {total}")

        # Chart
        chart_df = pd.DataFrame({
            "Category": ["Obtained", "Remaining"],
            "Marks": [obtained, total - obtained]
        })

        st.bar_chart(chart_df.set_index("Category"))
